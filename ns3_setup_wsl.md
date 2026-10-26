# ns-3 + 5G-LENA + ns3-gym setup (Windows laptop, WSL2)

This is the exact combination we built and tested in our Linux workspace
(Ubuntu 24.04, g++ 13.3, CMake 3.28, Python 3.11):

| Piece | Version | Where it comes from |
|---|---|---|
| ns-3 | **3.46** | `https://gitlab.com/nsnam/ns-3-dev`, tag `ns-3.46` |
| 5G-LENA (`nr` module) | **v4.1.1** | `https://gitlab.com/cttc-lena/nr` (GitHub mirror also works) |
| ns3-gym (`opengym` module) | branch **app-ns-3.36+** (commit cfff7f3, Mar 2025) | `https://github.com/tkn-tub/ns3-gym` |

5G-LENA v4.1 is the release made for ns-3.46. Do not mix versions: each 5G-LENA
release only builds against the ns-3 release it was made for.

Four small build fixes (step 4) and one Python fix (step 6) are needed. With
them, the RL-scheduler example below builds and runs end to end (checked on
our side: build + a 1 s simulation driven from Python).

---

## 1. Install WSL2 with Ubuntu 24.04 (once, in Windows PowerShell as admin)

```powershell
wsl --install -d Ubuntu-24.04
```
Restart when asked, open "Ubuntu 24.04" from the Start menu and create a user.
Keep all ns-3 files **inside** the Linux file system (`~/...`), not under
`/mnt/c/...`; building on the Windows drive is many times slower.

Give WSL enough memory: compiling 5G-LENA needs about 2 GB per compile job.
If the laptop has 16 GB, create `C:\Users\<you>\.wslconfig` with
```
[wsl2]
memory=10GB
processors=6
```
and run `wsl --shutdown` once.

## 2. System packages (inside Ubuntu)

```bash
sudo apt-get update
sudo apt-get install -y g++ cmake ninja-build git python3 python3-pip python3-venv \
    pkg-config libeigen3-dev libxml2-dev libsqlite3-dev \
    libzmq5 libzmq3-dev cppzmq-dev libprotobuf-dev protobuf-compiler
```
(Eigen is needed by 5G-LENA; ZeroMQ + protobuf are needed by ns3-gym.)

## 3. Get the sources

```bash
mkdir -p ~/ns3 && cd ~/ns3
git clone --branch ns-3.46 --depth 1 https://gitlab.com/nsnam/ns-3-dev.git ns-3.46
cd ns-3.46
git clone --branch v4.1.1 --depth 1 https://gitlab.com/cttc-lena/nr.git contrib/nr
git clone --branch app-ns-3.36+ https://github.com/tkn-tub/ns3-gym.git contrib/opengym
```

## 4. Four fixes (copy exactly)

```bash
cd ~/ns3/ns-3.46

# Fix 1: ns3-gym's example folder creates a program called "opengym", which
#        clashes with the library of the same name in ns-3.46.
mv contrib/opengym/examples contrib/opengym/examples_disabled

# Fix 2: ns-3.46 no longer creates the "<lib>-obj" helper target that
#        ns3-gym's CMake file asks for.
sed -i 's/TARGET \${libopengym-obj}/TARGET ${libopengym}/' contrib/opengym/CMakeLists.txt

# Fix 3: 5G-LENA v4.1.1 defines its AI-scheduler files (used by the RL example)
#        but never adds them to the library, so the example cannot find
#        nr-mac-scheduler-ai-ns3-gym-env.h.  Add them to the source/header lists.
sed -i 's/^    \${eigen_sources}$/    ${eigen_sources}\n    ${opengym_sources}/' contrib/nr/CMakeLists.txt
sed -i 's/^set(header_files$/set(header_files\n    ${opengym_headers}/' contrib/nr/CMakeLists.txt
grep -n 'opengym_sources}\|opengym_headers}' contrib/nr/CMakeLists.txt   # should print 2 lines

# Fix 4: ns-3.46 refuses Create<T>() for classes derived from ns3::Object
#        ("Use CreateObject() instead of Create() for Object subclasses").
sed -i 's/Create<OpenGymBoxSpace>/CreateObject<OpenGymBoxSpace>/' contrib/nr/model/nr-mac-scheduler-ai-ns3-gym-env.cc
```

## 5. Configure and build only what we need

```bash
./ns3 configure -d release --enable-examples --enable-modules=nr,opengym --disable-werror
# Build just the RL-scheduler example (and the libraries it needs), not all ~1000 targets:
cmake --build cmake-cache -j 4 --target gsoc-nr-rl-based-sched
```
`-j 4` = four compile jobs; lower it if WSL runs out of memory (the compiler
gets "Killed"). A full build of every 5G-LENA example is not needed.

## 6. Python side of ns3-gym

```bash
python3 -m venv ~/ns3/venv && source ~/ns3/venv/bin/activate
# Fix 5: ns3gym's Python code uses np.int / np.uint / np.float, which NumPy 2
#        removed ("module 'numpy' has no attribute 'float'").
sed -i -E 's/np\.int$/np.int64/; s/np\.uint$/np.uint64/; s/np\.float$/np.float64/' \
    contrib/opengym/model/ns3gym/ns3gym/ns3env.py
pip install ./contrib/opengym/model/ns3gym numpy torch
```

## 7. Smoke test

5G-LENA ships an example where a Python agent sets the scheduler's per-flow
weights over ns3-gym (`contrib/nr/examples/gsoc-nr-rl-based-sched/`).
The Python script starts the simulation itself. ns3-gym takes the program
name from the **name of the current folder**, so run it from the example folder:
```bash
cd ~/ns3/ns-3.46/contrib/nr/examples/gsoc-nr-rl-based-sched
python3 rl-sched-gym-env-intro.py
```
Expected output (what we got): `AI scheduler is enabled`, then lines such as
```
Step:  200
---action:  [1.9067005  0.70655499 1.71223721]
---obs, reward, done, info:  [ 2  5 25  8  2  4 68  8  1  4 68 11] 0.000256... False {}
```
and at the end a flow-monitor summary (Rx packets, mean throughput, delay).
Each group of four numbers in `obs` is one flow (UE id, logical channel, QoS
priority, head-of-line delay); `action` is one scheduling weight per flow.
If you see this, ns-3, 5G-LENA and ns3-gym are working together. The ignorable
"Gym has been unmaintained" and "Attribute 'RemotePort' is deprecated" warnings
are expected.

## How this fits our project

The Python simulator (`rra/env/pysim.py`) and the ns-3 bridge (Phase 6) both
implement the same `RRAEnv` interface (`rra/env/interface.py`). The ns-3 bridge
will map our per-user PRB shares onto the per-flow weights that this AI
scheduler already accepts, and read the per-user KPIs back. That is why we
picked this example as the smoke test.
