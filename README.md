# OpenEvolve

<div align="center">

<img src="openevolve-logo.png" alt="OpenEvolve Logo" width="400">

**🧬 The most advanced open-source evolutionary coding agent**

*Turn your LLMs into autonomous code optimizers that discover breakthrough algorithms*

<p align="center">
  <a href="https://github.com/codelion/openevolve/stargazers"><img src="https://img.shields.io/github/stars/codelion/openevolve?style=social" alt="GitHub stars"></a>
  <a href="https://pypi.org/project/openevolve/"><img src="https://img.shields.io/pypi/v/openevolve" alt="PyPI version"></a>
  <a href="https://pypi.org/project/openevolve/"><img src="https://img.shields.io/pypi/dm/openevolve" alt="PyPI downloads"></a>
  <a href="https://github.com/codelion/openevolve/blob/main/LICENSE"><img src="https://img.shields.io/github/license/codelion/openevolve" alt="License"></a>
</p>

[🚀 **Quick Start**](#-quick-start) • [📖 **Examples**](#-examples-gallery) • [📝 **System Messages**](#-crafting-effective-system-messages) • [💬 **Discussions**](https://github.com/codelion/openevolve/discussions)

*From random search to state-of-the-art: Watch your code evolve in real-time*

</div>

---

## ✨ Why OpenEvolve?

<table>
<tr>
<td width="33%">

### 🎯 **Autonomous Discovery**
LLMs don't just optimize—they **discover** entirely new algorithms. No human guidance needed.

</td>
<td width="33%">

### ⚡ **Proven Results** 
**2-3x speedups** on real hardware. **State-of-the-art** circle packing. **Breakthrough** optimizations.

</td>
<td width="33%">

### 🔬 **Research Grade**
Full reproducibility, extensive evaluation pipelines, and scientific rigor built-in.

</td>
</tr>
</table>

**OpenEvolve vs Manual Optimization:**

| Aspect | Manual Optimization | OpenEvolve |
|--------|-------------------|------------|
| **Time to Solution** | Days to weeks | Hours |
| **Exploration Breadth** | Limited by human creativity | Unlimited LLM creativity |
| **Reproducibility** | Hard to replicate | Fully deterministic |
| **Multi-objective** | Complex tradeoffs | Automatic Pareto optimization |
| **Scaling** | Doesn't scale | Parallel evolution across islands |

## 🏆 Proven Achievements

<div align="center">

| 🎯 **Domain** | 📈 **Achievement** | 🔗 **Example** |
|---------------|-------------------|----------------|
| **GPU Optimization** | 2-3x speedup on Apple Silicon | [MLX Metal Kernels](examples/mlx_metal_kernel_opt/) |
| **Mathematical** | State-of-the-art circle packing (n=26) | [Circle Packing](examples/circle_packing/) |
| **Algorithm Design** | Adaptive sorting algorithms | [Rust Adaptive Sort](examples/rust_adaptive_sort/) |
| **Scientific Computing** | Automated filter design | [Signal Processing](examples/signal_processing/) |
| **Multi-Language** | Python, Rust, R, Metal shaders | [All Examples](examples/) |

</div>

## 🚀 Quick Start

Get from zero to evolving code in **30 seconds**:

```bash
# Install OpenEvolve
pip install openevolve

# The example uses Google Gemini by default (free tier available)
# Get your API key from: https://aistudio.google.com/apikey
export OPENAI_API_KEY="your-gemini-api-key"  # Yes, use OPENAI_API_KEY env var

# Run your first evolution!
python openevolve-run.py examples/function_minimization/initial_program.py \
  examples/function_minimization/evaluator.py \
  --config examples/function_minimization/config.yaml \
  --iterations 50
```

**Note:** The example config uses Gemini by default, but you can use any OpenAI-compatible provider by modifying the `config.yaml`. See the [configs](configs/) for full configuration options.

### 📚 **Library Usage**

OpenEvolve can be used as a library without any external files:

```python
from openevolve import run_evolution, evolve_function

# Evolution with inline code (no files needed!)
result = run_evolution(
    initial_program='''
    def fibonacci(n):
        if n <= 1: return n
        return fibonacci(n-1) + fibonacci(n-2)
    ''',
    evaluator=lambda path: {"score": benchmark_fib(path)},
    iterations=100
)

# Evolve Python functions directly
def bubble_sort(arr):
    for i in range(len(arr)):
        for j in range(len(arr)-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j] 
    return arr

result = evolve_function(
    bubble_sort,
    test_cases=[([3,1,2], [1,2,3]), ([5,2,8], [2,5,8])],
    iterations=50
)
print(f"Evolved sorting algorithm: {result.best_code}")
```

**Prefer Docker?**
```bash
docker run --rm -v $(pwd):/app ghcr.io/codelion/openevolve:latest \
  examples/function_minimization/initial_program.py \
  examples/function_minimization/evaluator.py --iterations 100
```

## 🎬 See It In Action

<details>
<summary><b>🔥 Circle Packing: From Random to State-of-the-Art</b></summary>

**Watch OpenEvolve discover optimal circle packing in real-time:**

| Generation 1 | Generation 190 | Generation 460 (Final) |
|--------------|----------------|----------------------|
| ![Initial](examples/circle_packing/circle_packing_1.png) | ![Progress](examples/circle_packing/circle_packing_190.png) | ![Final](examples/circle_packing/circle_packing_460.png) |
| Random placement | Learning structure | **State-of-the-art result** |

**Result**: Matches published benchmarks for n=26 circle packing problem.

</details>

<details>
<summary><b>⚡ GPU Kernel Evolution</b></summary>

**Before (Baseline)**:
```metal
// Standard attention implementation
kernel void attention_baseline(/* ... */) {
    // Generic matrix multiplication
    float sum = 0.0;
    for (int i = 0; i < seq_len; i++) {
        sum += query[tid] * key[i];
    }
}
```

**After Evolution (2.8x faster)**:
```metal
// OpenEvolve discovered optimization
kernel void attention_evolved(/* ... */) {
    // Hardware-aware tiling + unified memory optimization
    threadgroup float shared_mem[256];
    // ... evolved algorithm exploiting Apple Silicon architecture
}
```

**Performance Impact**: 2.8x speedup on Apple M1 Pro, maintaining numerical accuracy.

</details>

## 🧬 How OpenEvolve Works

OpenEvolve implements a sophisticated **evolutionary coding pipeline** that goes far beyond simple optimization:

![OpenEvolve Architecture](openevolve-architecture.png)

### 🎯 **Core Innovation**: MAP-Elites + LLMs

- **Quality-Diversity Evolution**: Maintains diverse populations across feature dimensions
- **Island-Based Architecture**: Multiple populations prevent premature convergence
- **LLM Ensemble**: Multiple models with intelligent fallback strategies
- **Artifact Side-Channel**: Error feedback improves subsequent generations

### 🚀 **Advanced Features**

<details>
<summary><b>🔬 Scientific Reproducibility</b></summary>

- **Comprehensive Seeding**: Every component (LLM, database, evaluation) is seeded
- **Default Seed=42**: Immediate reproducible results out of the box
- **Deterministic Evolution**: Exact reproduction of runs across machines
- **Component Isolation**: Hash-based isolation prevents cross-contamination

</details>

<details>
<summary><b>🤖 Advanced LLM Integration</b></summary>

- **Test-Time Compute**: Integration with [OptiLLM](https://github.com/codelion/optillm) for MoA and enhanced reasoning
- **Universal API**: Works with OpenAI, Google, local models
- **Plugin Ecosystem**: Support for OptiLLM plugins (readurls, executecode, z3_solver)
- **Intelligent Ensembles**: Weighted combinations with sophisticated fallback

</details>

<details>
<summary><b>🧬 Evolution Algorithm Innovations</b></summary>

- **Double Selection**: Different programs for performance vs inspiration
- **Adaptive Feature Dimensions**: Custom quality-diversity metrics
- **Migration Patterns**: Ring topology with controlled gene flow
- **Multi-Strategy Sampling**: Elite, diverse, and exploratory selection

</details>

## 🎯 Perfect For

| **Use Case** | **Why OpenEvolve Excels** |
|--------------|---------------------------|
| 🏃‍♂️ **Performance Optimization** | Discovers hardware-specific optimizations humans miss |
| 🧮 **Algorithm Discovery** | Finds novel approaches to classic problems |
| 🔬 **Scientific Computing** | Automates tedious manual tuning processes |
| 🎮 **Competitive Programming** | Generates multiple solution strategies |
| 📊 **Multi-Objective Problems** | Pareto-optimal solutions across dimensions |

## 🛠 Installation & Setup

### Requirements
- **Python**: 3.10+ 
- **LLM Access**: Any OpenAI-compatible API
- **Optional**: Docker for containerized runs

### Installation Options

<details>
<summary><b>📦 PyPI (Recommended)</b></summary>

```bash
pip install openevolve
```

</details>

<details>
<summary><b>🔧 Development Install</b></summary>

```bash
git clone https://github.com/codelion/openevolve.git
cd openevolve
pip install -e ".[dev]"
```

</details>

<details>
<summary><b>🐳 Docker</b></summary>

```bash
docker pull ghcr.io/codelion/openevolve:latest
```

</details>

### LLM Provider Setup

OpenEvolve works with **any OpenAI-compatible API**:

<details>
<summary><b>🔥 OpenAI (Direct)</b></summary>

```bash
export OPENAI_API_KEY="sk-..."
# Uses OpenAI endpoints by default
```

</details>

<details>
<summary><b>🤖 Google Gemini</b></summary>

```yaml
# config.yaml
llm:
  api_base: "https://generativelanguage.googleapis.com/v1beta/openai/"
  model: "gemini-2.5-pro"
```

```bash
export OPENAI_API_KEY="your-gemini-api-key"
```

</details>

<details>
<summary><b>🏠 Local Models (Ollama/vLLM)</b></summary>

```yaml
# config.yaml
llm:
  api_base: "http://localhost:11434/v1"  # Ollama
  model: "codellama:7b"
```

</details>

<details>
<summary><b>⚡ OptiLLM (Advanced)</b></summary>

For maximum flexibility with rate limiting, model routing, and test-time compute:

```bash
# Install OptiLLM
pip install optillm

# Start OptiLLM proxy
optillm --port 8000

# Point OpenEvolve to OptiLLM
export OPENAI_API_KEY="your-actual-key"
```

```yaml
llm:
  api_base: "http://localhost:8000/v1"
  model: "moa&readurls-o3"  # Test-time compute + web access
```

</details>

## 📸 Examples Gallery

<div align="center">

### 🏆 **Showcase Projects**

| Project | Domain | Achievement | Demo |
|---------|--------|-------------|------|
| [🎯 **Function Minimization**](examples/function_minimization/) | Optimization | Random → Simulated Annealing | [View Results](examples/function_minimization/openevolve_output/) |
| [⚡ **MLX GPU Kernels**](examples/mlx_metal_kernel_opt/) | Hardware | 2-3x Apple Silicon speedup | [Benchmarks](examples/mlx_metal_kernel_opt/README.md) |
| [🔄 **Rust Adaptive Sort**](examples/rust_adaptive_sort/) | Algorithms | Data-aware sorting | [Code Evolution](examples/rust_adaptive_sort/) |
| [📐 **Symbolic Regression**](examples/symbolic_regression/) | Science | Automated equation discovery | [LLM-SRBench](examples/symbolic_regression/) |
| [🕸️ **Web Scraper + OptiLLM**](examples/web_scraper_optillm/) | AI Integration | Test-time compute optimization | [Smart Scraping](examples/web_scraper_optillm/) |

</div>

### 🎯 **Quick Example**: Function Minimization

**Watch OpenEvolve evolve from random search to sophisticated optimization:**

```python
# Initial Program (Random Search)
def minimize_function(func, bounds, max_evals=1000):
    best_x, best_val = None, float('inf')
    for _ in range(max_evals):
        x = random_point_in_bounds(bounds)
        val = func(x)
        if val < best_val:
            best_x, best_val = x, val
    return best_x, best_val
```

**↓ Evolution Process ↓**

```python
# Evolved Program (Simulated Annealing + Adaptive Cooling)
def minimize_function(func, bounds, max_evals=1000):
    x = random_point_in_bounds(bounds)
    temp = adaptive_initial_temperature(func, bounds)
    
    for i in range(max_evals):
        neighbor = generate_neighbor(x, temp, bounds)
        delta = func(neighbor) - func(x)
        
        if delta < 0 or random.random() < exp(-delta/temp):
            x = neighbor
            
        temp *= adaptive_cooling_rate(i, max_evals)  # Dynamic cooling
    
    return x, func(x)
```

**Performance**: 100x improvement in convergence speed!

### 🔬 **Advanced Examples**

<details>
<summary><b>🎨 Prompt Evolution</b></summary>

**Evolve prompts instead of code** for better LLM performance:

```yaml
# Example: HotpotQA dataset
Initial Prompt: "Answer the question based on the context."

Evolved Prompt: "As an expert analyst, carefully examine the provided context. 
Break down complex multi-hop reasoning into clear steps. Cross-reference 
information from multiple sources to ensure accuracy. Answer: [question]"

Result: +23% accuracy improvement on HotpotQA benchmark
```

[Full Example](examples/llm_prompt_optimization/)

</details>

<details>
<summary><b>🏁 Competitive Programming</b></summary>

**Automatic solution generation** for programming contests:

```python
# Problem: Find maximum subarray sum
# OpenEvolve discovers multiple approaches:

# Evolution Path 1: Brute Force → Kadane's Algorithm
# Evolution Path 2: Divide & Conquer → Optimized Kadane's
# Evolution Path 3: Dynamic Programming → Space-Optimized DP
```

[Online Judge Integration](examples/online_judge_programming/)

</details>

## ⚙️ Configuration

OpenEvolve offers extensive configuration for advanced users:

```yaml
# Advanced Configuration Example
max_iterations: 1000
random_seed: 42  # Full reproducibility

llm:
  # Ensemble with test-time compute
  models:
    - name: "gemini-2.5-pro"
      weight: 0.6
    - name: "moa&readurls-o3"  # OptiLLM features
      weight: 0.4
  temperature: 0.7

database:
  # MAP-Elites quality-diversity
  population_size: 500
  num_islands: 5  # Parallel evolution
  migration_interval: 20
  feature_dimensions: ["complexity", "diversity", "performance"]

evaluator:
  enable_artifacts: true      # Error feedback to LLM
  cascade_evaluation: true    # Multi-stage testing
  use_llm_feedback: true      # AI code quality assessment

prompt:
  # Sophisticated inspiration system
  num_top_programs: 3         # Best performers
  num_diverse_programs: 2     # Creative exploration
  include_artifacts: true     # Execution feedback
  
  # Custom templates
  template_dir: "custom_prompts/"
  use_template_stochasticity: true  # Randomized prompts
```

<details>
<summary><b>🎯 Feature Engineering</b></summary>

**Control how programs are organized in the quality-diversity grid:**

```yaml
database:
  feature_dimensions: 
    - "complexity"      # Built-in: code length
    - "diversity"       # Built-in: structural diversity
    - "performance"     # Custom: from your evaluator
    - "memory_usage"    # Custom: from your evaluator
    
  feature_bins:
    complexity: 10      # 10 complexity levels
    performance: 20     # 20 performance buckets
    memory_usage: 15    # 15 memory usage categories
```

**Important**: Return raw values from evaluator, OpenEvolve handles binning automatically.

</details>

<details>
<summary><b>🎨 Custom Prompt Templates</b></summary>

**Advanced prompt engineering** with custom templates:

```yaml
prompt:
  template_dir: "custom_templates/"
  use_template_stochasticity: true
  template_variations:
    greeting:
      - "Let's enhance this code:"
      - "Time to optimize:"
      - "Improving the algorithm:"
```

See [prompt examples](examples/llm_prompt_optimization/templates/) for complete template customization.

</details>

## 📝 Crafting Effective System Messages

**System messages are the secret to successful evolution.** They guide the LLM's understanding of your domain, constraints, and optimization goals. A well-crafted system message can be the difference between random mutations and targeted improvements.

### Why System Messages Matter

The system message in your config.yaml is arguably the most important component for evolution success:

- **Domain Expertise**: Provides LLM with specific knowledge about your problem space
- **Constraint Awareness**: Defines what can and cannot be changed during evolution
- **Optimization Focus**: Guides the LLM toward meaningful improvements
- **Error Prevention**: Helps avoid common pitfalls and compilation errors

### The Iterative Creation Process

Based on successful OpenEvolve implementations, system messages are best created through iteration:

<details>
<summary><b>🔄 Step-by-Step Process</b></summary>

**Phase 1: Initial Draft**

1. Start with a basic system message describing your goal
2. Run 20-50 evolution iterations to observe behavior
3. Note where the system gets "stuck" or makes poor choices

**Phase 2: Refinement**

4. Add specific guidance based on observed issues
5. Include domain-specific terminology and concepts
6. Define clear constraints and optimization targets
7. Run another batch of iterations

**Phase 3: Specialization**

8. Add detailed examples of good vs bad approaches
9. Include specific library/framework guidance
10. Add error avoidance patterns you've observed
11. Fine-tune based on artifact feedback

**Phase 4: Optimization**

12. Consider using OpenEvolve itself to optimize your prompt
13. Measure improvements using combined score metrics

</details>

### Examples by Complexity

#### 🎯 **Simple: General Optimization**
```yaml
prompt:
  system_message: |
    You are an expert programmer specializing in optimization algorithms.
    Your task is to improve a function minimization algorithm to find the
    global minimum reliably, escaping local minima that might trap simple algorithms.
```

#### 🔧 **Intermediate: Domain-Specific Guidance**
```yaml
prompt:
  system_message: |
    You are an expert prompt engineer. Your task is to revise prompts for LLMs.

    Your improvements should:
    * Clarify vague instructions and eliminate ambiguity
    * Strengthen alignment between prompt and desired task outcome
    * Improve robustness against edge cases
    * Include formatting instructions and examples where helpful
    * Avoid unnecessary verbosity

    Return only the improved prompt text without explanations.
```

#### ⚡ **Advanced: Hardware-Specific Optimization**
```yaml
prompt:
  system_message: |
    You are an expert Metal GPU programmer specializing in custom attention
    kernels for Apple Silicon.

    # TARGET: Optimize Metal Kernel for Grouped Query Attention (GQA)
    # HARDWARE: Apple M-series GPUs with unified memory architecture
    # GOAL: 5-15% performance improvement

    # OPTIMIZATION OPPORTUNITIES:
    **1. Memory Access Pattern Optimization:**
    - Coalesced access patterns for Apple Silicon
    - Vectorized loading using SIMD
    - Pre-compute frequently used indices

    **2. Algorithm Fusion:**
    - Combine max finding with score computation
    - Reduce number of passes through data

    # CONSTRAINTS - CRITICAL SAFETY RULES:
    **MUST NOT CHANGE:**
    ❌ Kernel function signature
    ❌ Template parameter names or types
    ❌ Overall algorithm correctness

    **ALLOWED TO OPTIMIZE:**
    ✅ Memory access patterns and indexing
    ✅ Computation order and efficiency
    ✅ Vectorization and SIMD utilization
    ✅ Apple Silicon specific optimizations
```

### Best Practices

<details>
<summary><b>🎨 Prompt Engineering Patterns</b></summary>

**Structure Your Message:**

- Start with role definition ("You are an expert...")
- Define the specific task and context
- List optimization opportunities with examples
- Set clear constraints and safety rules
- End with success criteria

**Use Specific Examples:**

```yaml
# Good: Specific optimization targets
system_message: |
  Focus on reducing memory allocations in the hot loop.
  Example: Replace `new Vector()` with pre-allocated arrays.

# Avoid: Vague guidance
system_message: "Make the code faster"
```

**Include Domain Knowledge:**

```yaml
# Good: Domain-specific guidance
system_message: |
  For GPU kernels, prioritize:
  1. Memory coalescing (access patterns)
  2. Occupancy (thread utilization)
  3. Shared memory usage (cache blocking)

# Avoid: Generic optimization advice
system_message: "Optimize the algorithm"
```

**Set Clear Boundaries:**

```yaml
system_message: |
  MUST NOT CHANGE:
  ❌ Function signatures
  ❌ Algorithm correctness
  ❌ External API compatibility

  ALLOWED TO OPTIMIZE:
  ✅ Internal implementation details
  ✅ Data structures and algorithms
  ✅ Performance optimizations
```

</details>

<details>
<summary><b>🔬 Advanced Techniques</b></summary>

**Artifact-Driven Iteration:**

- Enable artifacts in your config
- Include common error patterns in system message
- Add guidance based on stderr/warning patterns

**Multi-Phase Evolution:**

```yaml
# Phase 1: Broad exploration
system_message: "Explore different algorithmic approaches..."

# Phase 2: Focused optimization
system_message: "Given the successful simulated annealing approach,
focus on parameter tuning and cooling schedules..."
```

**Template Variation:**

```yaml
prompt:
  template_dir: "custom_templates/"
  use_template_stochasticity: true
  system_message: |
    # Use multiple greeting variations
    [Randomly: "Let's optimize this code:" | "Time to enhance:" | "Improving:"]
```

</details>

### Meta-Evolution: Using OpenEvolve to Optimize Prompts

**You can use OpenEvolve to evolve your system messages themselves!**

```yaml
# Example: Evolve prompts for HotpotQA dataset
Initial Prompt: "Answer the question based on the context."

Evolved Prompt: "As an expert analyst, carefully examine the provided context.
Break down complex multi-hop reasoning into clear steps. Cross-reference
information from multiple sources to ensure accuracy. Answer: [question]"

Result: +23% accuracy improvement on HotpotQA benchmark
```

See the [LLM Prompt Optimization example](examples/llm_prompt_optimization/) for a complete implementation.

### Common Pitfalls to Avoid

- **Too Vague**: "Make the code better" → Specify exactly what "better" means
- **Too Restrictive**: Over-constraining can prevent useful optimizations
- **Missing Context**: Include relevant domain knowledge and terminology
- **No Examples**: Concrete examples guide LLM better than abstract descriptions
- **Ignoring Artifacts**: Don't refine prompts based on error feedback

## 🔧 Artifacts & Debugging

**Artifacts side-channel** provides rich feedback to accelerate evolution:

```python
# Evaluator can return execution context
from openevolve.evaluation_result import EvaluationResult

return EvaluationResult(
    metrics={"performance": 0.85, "correctness": 1.0},
    artifacts={
        "stderr": "Warning: suboptimal memory access pattern",
        "profiling_data": {...},
        "llm_feedback": "Code is correct but could use better variable names",
        "build_warnings": ["unused variable x"]
    }
)
```

**Next generation prompt automatically includes:**

```markdown
## Previous Execution Feedback
⚠️ Warning: suboptimal memory access pattern
💡 LLM Feedback: Code is correct but could use better variable names
🔧 Build Warnings: unused variable x
```

This creates a **feedback loop** where each generation learns from previous mistakes!

## 📊 Visualization

**Real-time evolution tracking** with interactive web interface:

```bash
# Install visualization dependencies
pip install -r scripts/requirements.txt

# Launch interactive visualizer
python scripts/visualizer.py

# Or visualize specific checkpoint
python scripts/visualizer.py --path examples/function_minimization/openevolve_output/checkpoints/checkpoint_100/
```

**Features:**

- 🌳 **Evolution tree** with parent-child relationships
- 📈 **Performance tracking** across generations
- 🔍 **Code diff viewer** showing mutations
- 📊 **MAP-Elites grid** visualization
- 🎯 **Multi-metric analysis** with custom dimensions

![OpenEvolve Visualizer](openevolve-visualizer.png)

## 🚀 Roadmap

### **🔥 Upcoming Features**

- [ ] **Multi-Modal Evolution**: Images, audio, and text simultaneously
- [ ] **Federated Learning**: Distributed evolution across multiple machines  
- [ ] **AutoML Integration**: Hyperparameter and architecture evolution
- [ ] **Benchmark Suite**: Standardized evaluation across domains

### **🌟 Research Directions**

- [ ] **Self-Modifying Prompts**: Evolution modifies its own prompting strategy
- [ ] **Cross-Language Evolution**: Python → Rust → C++ optimization chains
- [ ] **Neurosymbolic Reasoning**: Combine neural and symbolic approaches
- [ ] **Human-AI Collaboration**: Interactive evolution with human feedback

Want to contribute? Check out our [roadmap discussions](https://github.com/codelion/openevolve/discussions/categories/roadmap)!

## 🤔 FAQ

<details>
<summary><b>💰 How much does it cost to run?</b></summary>

**Cost depends on your LLM provider and iterations:**

- **o3**: ~$0.15-0.60 per iteration (depending on code size)
- **o3-mini**: ~$0.03-0.12 per iteration (more cost-effective)
- **Gemini-2.5-Pro**: ~$0.08-0.30 per iteration
- **Gemini-2.5-Flash**: ~$0.01-0.05 per iteration (fastest and cheapest)
- **Local models**: Nearly free after setup
- **OptiLLM**: Use cheaper models with test-time compute for better results

**Cost-saving tips:**
- Start with fewer iterations (100-200)
- Use o3-mini, Gemini-2.5-Flash or local models for exploration
- Use cascade evaluation to filter bad programs early
- Configure smaller population sizes initially

</details>

<details>
<summary><b>🆚 How does this compare to manual optimization?</b></summary>

| Aspect | Manual | OpenEvolve |
|--------|--------|------------|
| **Initial Learning** | Weeks to understand domain | Minutes to start |
| **Solution Quality** | Depends on expertise | Consistently explores novel approaches |
| **Time Investment** | Days-weeks per optimization | Hours for complete evolution |
| **Reproducibility** | Hard to replicate exact process | Perfect reproduction with seeds |
| **Scaling** | Doesn't scale beyond human capacity | Parallel evolution across islands |

**OpenEvolve shines** when you need to explore large solution spaces or optimize for multiple objectives simultaneously.

</details>

<details>
<summary><b>🔧 Can I use my own LLM?</b></summary>

**Yes!** OpenEvolve supports any OpenAI-compatible API:

- **Commercial**: OpenAI, Google, Cohere
- **Local**: Ollama, vLLM, LM Studio, text-generation-webui
- **Advanced**: OptiLLM for routing and test-time compute

Just set the `api_base` in your config to point to your endpoint.

</details>

<details>
<summary><b>🚨 What if evolution gets stuck?</b></summary>

**Built-in mechanisms prevent stagnation:**

- **Island migration**: Fresh genes from other populations
- **Temperature control**: Exploration vs exploitation balance
- **Diversity maintenance**: MAP-Elites prevents convergence
- **Artifact feedback**: Error messages guide improvements
- **Template stochasticity**: Randomized prompts break patterns

**Manual interventions:**
- Increase `num_diverse_programs` for more exploration
- Add custom feature dimensions to diversify search
- Use template variations to randomize prompts
- Adjust migration intervals for more cross-pollination

</details>

<details>
<summary><b>📈 How do I measure success?</b></summary>

**Multiple success metrics:**

1. **Primary Metric**: Your evaluator's `combined_score` or metric average
2. **Convergence**: Best score improvement over time
3. **Diversity**: MAP-Elites grid coverage
4. **Efficiency**: Iterations to reach target performance
5. **Robustness**: Performance across different test cases

**Use the visualizer** to track all metrics in real-time and identify when evolution has converged.

</details>

### 🌟 **Contributors**

Thanks to all our amazing contributors who make OpenEvolve possible!

<a href="https://github.com/codelion/openevolve/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=codelion/openevolve" />
</a>

### 🤝 **Contributing**

We welcome contributions! Here's how to get started:

1. 🍴 **Fork** the repository
2. 🌿 **Create** your feature branch: `git checkout -b feat-amazing-feature`
3. ✨ **Add** your changes and tests
4. ✅ **Test** everything: `python -m unittest discover tests`
5. 📝 **Commit** with a clear message
6. 🚀 **Push** and create a Pull Request

**New to open source?** Check out our [Contributing Guide](CONTRIBUTING.md) and look for [`good-first-issue`](https://github.com/codelion/openevolve/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) labels!

### 📚 **Academic & Research**

**Articles & Blog Posts About OpenEvolve**:
- [Towards Open Evolutionary Agents](https://huggingface.co/blog/driaforall/towards-open-evolutionary-agents) - Evolution of coding agents and the open-source movement
- [OpenEvolve: GPU Kernel Discovery](https://huggingface.co/blog/codelion/openevolve-gpu-kernel-discovery) - Automated discovery of optimized GPU kernels with 2-3x speedups
- [OpenEvolve: Evolutionary Coding with LLMs](https://huggingface.co/blog/codelion/openevolve) - Introduction to evolutionary algorithm discovery using large language models

## 📊 Citation

If you use OpenEvolve in your research, please cite:

```bibtex
@software{openevolve,
  title = {OpenEvolve: an open-source evolutionary coding agent},
  author = {Asankhaya Sharma},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/codelion/openevolve}
}
```
---

<div align="center">

### **🚀 Ready to evolve your code?**

**Made with ❤️ by the OpenEvolve community**

*Star ⭐ this repository if OpenEvolve helps you discover breakthrough algorithms!*

</div>
