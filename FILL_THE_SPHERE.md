# HOW TO FILL THE SPHERE

You have three tools to work with:

## 1. FILL THE SPHERE (Simplest - Just Let It Run)

```bash
# Run forever until you stop it (Ctrl+C)
python fill_sphere.py --quiet

# Run for specific number of sessions
python fill_sphere.py --sessions 1000 --quiet

# Verbose mode (see every reasoning step)
python fill_sphere.py --sessions 100
```

**What it does:**
- Picks random concept pairs
- Attempts transitions between them
- Records successful pathways
- Strengthens pathways with each use
- Keeps going until you stop it

**Output every 10 sessions:**
```
Session   10 | Pathways:    5 | Avg strength: 0.169 | Rate: 7.5/s
Session   20 | Pathways:    9 | Avg strength: 0.133 | Rate: 7.5/s
```

## 2. EXPLORE THE SPHERE (See What You've Created)

```bash
python explore_concept_space.py
```

**Interactive commands:**
```
show understand              # See concept details + polarity signature
compare understand interpret # Compare two concepts
neighborhood understand 0.5  # Find nearby concepts
network                      # Analyze pathway structure
cloud                        # List all concepts
quit                         # Exit
```

**What you see:**
- **Position** - Where concept lives on Bloch sphere (θ, φ)
- **Polarity signature** - How concept is scored (5 axes)
- **Nearest neighbors** - Similar concepts by distance
- **Pathways** - Roads built to/from this concept
- **Network stats** - Hubs, attractors, strongest paths

## 3. AUTONOMOUS EXPLORER (Advanced)

```bash
# Random exploration
python autonomous_explorer.py --strategy random --sessions 100

# Frontier exploration (explores from known → unknown)
python autonomous_explorer.py --strategy frontier --sessions 500

# Dense exploration (strengthens existing pathways)
python autonomous_explorer.py --strategy dense --sessions 1000

# Time-based (run for 5 minutes)
python autonomous_explorer.py --strategy frontier --time 300
```

**Strategies:**
- `random` - Pure random concept pairs
- `frontier` - Start from known hubs, explore to unexplored targets
- `dense` - 70% strengthen existing paths, 30% explore new

---

## WORKFLOW

### Phase 1: Initial Exploration (Build the Roads)
```bash
# Let it run for a while (Ctrl+C when ready)
python fill_sphere.py --quiet
```

This runs sessions at ~7-8 per second, building pathways.

After 1000 sessions, you'll have ~200-300 pathways built.

### Phase 2: Inspect What You've Created
```bash
python explore_concept_space.py
```

Commands to try:
```
network                    # See overall structure
show compare               # Inspect a concept
compare compare differentiate
```

### Phase 3: Strengthen Important Pathways
```bash
# Run dense strategy to strengthen high-use roads
python autonomous_explorer.py --strategy dense --sessions 500 --report 50
```

### Phase 4: Keep Going
```bash
# Just keep filling forever
python fill_sphere.py --quiet
```

Leave it running overnight → thousands of pathways

---

## WHAT YOU'RE BUILDING

**Concepts:** 276 total
- 4 base primitives: compare, interpret, generate, select
- 16 gen-1 pairs: compare_interpret, interpret_generate, etc.
- 256 gen-2 pairs: understand_evaluate, etc.

**Pathways:** Initially 0, grows with exploration
- Each pathway has: usage count, success rate, strength (0-1)
- Strength > 0.7 → promoted to "known road"
- High-strength pathways → faster convergence

**Scoring:** Each concept scored on 5 axes
- existence: abstract ← → concrete
- gender: feminine ← → masculine
- binary: continuous ← → discrete
- temperature: hot/fast ← → cool/slow
- density: simple ← → complex

**Coverage:**
- Possible transitions: 276 × 276 = 76,176
- After 1000 sessions: ~300 pathways = 0.4% coverage
- After 10,000 sessions: ~2000 pathways = 2.6% coverage
- After 100,000 sessions: ~15,000 pathways = 20% coverage

The sphere fills slowly but cumulatively - each pathway is permanent infrastructure.

---

## TIPS

**Speed:**
- Quiet mode (`--quiet`) runs ~7-8 sessions/sec
- Verbose mode runs ~1-2 sessions/sec (lots of output)
- Each session = one reasoning problem solved

**Quality:**
- Frontier strategy finds diverse pathways
- Dense strategy strengthens important ones
- Random strategy explores uniformly

**State:**
- Everything saves to `learning_engine_state.json`
- Safe to stop/restart anytime (loads previous state)
- Pathways accumulate across runs

**Monitoring:**
- Use `explore_concept_space.py` to inspect progress
- `network` command shows overall structure
- Watch for average strength increasing over time

---

## EXAMPLE: OVERNIGHT RUN

```bash
# Start before bed
nohup python fill_sphere.py --quiet > sphere_log.txt 2>&1 &

# Check progress next morning
tail sphere_log.txt

# Explore what was built
python explore_concept_space.py
```

After 8 hours at 7 sessions/sec:
- 8 × 3600 × 7 = 201,600 sessions
- ~10,000+ pathways built
- ~13% of sphere covered
- Many pathways with strength > 0.7 (instant snaps)

**The sphere is getting DENSE.**
