# My Test Codes: Experimental & Development Scripts

## Overview
This folder contains **experimental, development-stage, and test scripts** used for:
- Prototyping new analysis approaches
- Debugging workflows
- Testing data processing pipelines
- Exploratory analysis
- Validation and verification

## Status
⚠️ **These scripts are NOT production-ready.** They may:
- Have incomplete implementations
- Lack error handling
- Use hardcoded paths
- Be abandoned/deprecated
- Require manual parameter adjustment

## Folder Contents

```
my test codes/
├── bound_AB.py                      # Boundary detection test
├── CSVtoPickle_Tortuous.py          # Tortuous PKL generation (test)
├── Graph_analysis.ipynb             # Analysis experimentation notebook
└── README.md
```

## Scripts

### 1. **bound_AB.py** - Boundary Detection Experimentation
- **Purpose:** Test boundary/interface detection algorithms
- **Topic:** Identifies edges crossing anatomical boundaries
- **Status:** Experimental - may not be used in production
- **Dependencies:** graph_analysis_functions, numpy, pickle
- **Typical use:** `python bound_AB.py` (check hardcoded paths first)

### 2. **CSVtoPickle_Tortuous.py** - Tortuous Path Conversion (Test)
- **Purpose:** Alternative implementation for tortuous geometry PKL generation
- **Comparison:** Different approach to [CSVtoPKL/CSVtoPKL_Tortuous_OutGeom.py](../CSVtoPKL/README.md)
- **Status:** Developmental - compare results with main implementation
- **Key features (to verify):**
  - CSV → PKL conversion with tortuous paths
  - Per-segment length computation
  - Tortuosity calculations
- **Usage:** Review and compare with production version before adoption

### 3. **Graph_analysis.ipynb** - Interactive Analysis Notebook
- **Purpose:** Explore graph analysis concepts and prototype metrics
- **Audience:** Developers experimenting with new metrics
- **Content:**
  - Data loading and exploration
  - Various metric computations
  - Visualization attempts
  - Debugging workflows
- **Status:** Working notebook - cell ordering may not be optimal

## How These Relate to Production Code

```
Production workflow:
  CSVtoPKL → cutting → Graph Analysis & by region → PKLtoVTP

Test/Dev folder:
  ├─ bound_AB.py ..................... → potential new boundary metric
  ├─ CSVtoPickle_Tortuous.py ......... → alternative to CSVtoPKL/CSVtoPKL_Tortuous_OutGeom.py
  └─ Graph_analysis.ipynb ............ → prototype for Graph Analysis & by region
```

## Using These Scripts Safely

**If you want to try a test script:**

1. **Make a copy** to avoid losing original
   ```bash
   cp bound_AB.py bound_AB_backup.py
   ```

2. **Check hardcoded paths**
   ```python
   # Look for lines like:
   # in_path = "/home/admin/Ana/MicroBrain/output/..."
   # Modify to your actual data locations
   ```

3. **Test on a small dataset first**
   - Use cut/subset graphs
   - Keep output in temporary folder
   - Validate results manually

4. **Compare with production equivalent**
   - If production code exists, compare outputs
   - Check for numerical differences
   - Validate data structures match

## Common Issues with Test Code

| Issue | Solution |
|-------|----------|
| `FileNotFoundError` | Update hardcoded paths to your data |
| Missing imports | Install required packages (igraph, numpy, etc.) |
| Index/shape errors | Check if data structure matches assumptions |
| NaN/Inf values | Verify input data validity, check threshold parameters |
| Performance issues | Test code may not be optimized; consider production version |

## When to Promote to Production

Move a test script to production:
- ✓ Verify correctness against known results
- ✓ Add proper error handling and logging
- ✓ Parameterize hardcoded values
- ✓ Add docstrings and comments
- ✓ Test on multiple datasets
- ✓ Remove debug print statements
- ✓ Compare with similar production code
- ✓ Add to appropriate module folder

### Example Promotion Path
```
my test codes/
  CSVtoPickle_Tortuous.py
          ↓
          ↓ (after validation)
          ↓
CSVtoPKL/
  CSVtoPickle_Tortuous_v2.py
```

## Maintenance

**Rules for this folder:**
- Scripts can be incomplete or incorrect ✓
- No guarantee of working state ✓
- May be deleted without notice ✓ (but shouldn't be!)
- Should document what you're testing ✓
- Include comments about status/known issues ✓

**Before using any test script:**
1. Understand its purpose
2. Verify it actually works on sample data
3. Compare outputs with production equivalent if one exists
4. Don't rely on it for critical analysis without validation

## Documentation in This Folder

When adding new test scripts, include:
1. **Purpose:** What are you testing/developing?
2. **Status:** Working / Incomplete / Deprecated?
3. **Known issues:** What's broken or incomplete?
4. **How to test:** Example commands and validation
5. **Related production code:** Which module should this become?

---

**Note:** This folder is for experimentation. For reliable, production-grade analysis, use the main modules in the parent `codes/` directory.

## Author
Ana Barrio - Feb 2026
