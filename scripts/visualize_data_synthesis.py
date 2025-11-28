#!/usr/bin/env python3
"""
Visualize the Data Synthesis behavior in Agentic CPT
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.lines as mlines

# Create figure with high DPI for clarity
fig, ax = plt.subplots(1, 1, figsize=(16, 12), dpi=100)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(5, 9.5, 'Agentic CPT: Data Synthesis Framework',
        fontsize=20, fontweight='bold', ha='center')
ax.text(5, 9.0, 'Complete Agent Workflow Lifecycle',
        fontsize=14, ha='center', style='italic', color='gray')

# Color scheme
color_memory = '#E8F4F8'
color_question = '#FFE5B4'
color_planning = '#B4E5FF'
color_reasoning = '#C5E1A5'
color_decision = '#FFCCBC'
color_env = '#E1BEE7'
color_agent_cycle = '#FFF9C4'

# ============ SECTION 1: Entity-Anchored Open-World Memory (Top) ============
memory_box = FancyBboxPatch((0.5, 7.5), 9, 1.2,
                            boxstyle="round,pad=0.1",
                            edgecolor='#0288D1', linewidth=2.5,
                            facecolor=color_memory)
ax.add_patch(memory_box)
ax.text(5, 8.4, 'Entity-Anchored Open-World Memory',
        fontsize=13, fontweight='bold', ha='center')
ax.text(5, 8.0, 'Web-crawled data + Agent interaction trajectories → Structured entity knowledge',
        fontsize=10, ha='center', style='italic')

# ============ SECTION 2: Four Data Synthesis Types (Middle) ============
y_synthesis = 5.5

# 1. Question Synthesis
question_box = FancyBboxPatch((0.3, y_synthesis), 2, 1.5,
                              boxstyle="round,pad=0.08",
                              edgecolor='#F57C00', linewidth=2,
                              facecolor=color_question)
ax.add_patch(question_box)
ax.text(1.3, y_synthesis + 1.2, '1. Question', fontsize=11, fontweight='bold', ha='center')
ax.text(1.3, y_synthesis + 1.0, 'Synthesis', fontsize=11, fontweight='bold', ha='center')
ax.text(1.3, y_synthesis + 0.65, 'Sample entities', fontsize=8, ha='center')
ax.text(1.3, y_synthesis + 0.45, '+ knowledge', fontsize=8, ha='center')
ax.text(1.3, y_synthesis + 0.25, '↓', fontsize=10, ha='center')
ax.text(1.3, y_synthesis + 0.05, 'Multi-style questions', fontsize=8, ha='center')

# 2. Planning Action
planning_box = FancyBboxPatch((2.6, y_synthesis), 2, 1.5,
                              boxstyle="round,pad=0.08",
                              edgecolor='#0288D1', linewidth=2,
                              facecolor=color_planning)
ax.add_patch(planning_box)
ax.text(3.6, y_synthesis + 1.2, '2. Planning', fontsize=11, fontweight='bold', ha='center')
ax.text(3.6, y_synthesis + 1.0, 'Action', fontsize=11, fontweight='bold', ha='center')
ax.text(3.6, y_synthesis + 0.65, 'Problem', fontsize=8, ha='center')
ax.text(3.6, y_synthesis + 0.45, 'decomposition', fontsize=8, ha='center')
ax.text(3.6, y_synthesis + 0.25, '↓', fontsize=10, ha='center')
ax.text(3.6, y_synthesis + 0.05, 'First-step action', fontsize=8, ha='center')

# 3. Reasoning Action
reasoning_box = FancyBboxPatch((4.9, y_synthesis), 2, 1.5,
                               boxstyle="round,pad=0.08",
                               edgecolor='#689F38', linewidth=2,
                               facecolor=color_reasoning)
ax.add_patch(reasoning_box)
ax.text(5.9, y_synthesis + 1.2, '3. Reasoning', fontsize=11, fontweight='bold', ha='center')
ax.text(5.9, y_synthesis + 1.0, 'Action', fontsize=11, fontweight='bold', ha='center')
ax.text(5.9, y_synthesis + 0.65, 'Knowledge', fontsize=8, ha='center')
ax.text(5.9, y_synthesis + 0.45, 'integration', fontsize=8, ha='center')
ax.text(5.9, y_synthesis + 0.25, '↓', fontsize=10, ha='center')
ax.text(5.9, y_synthesis + 0.05, 'Reasoning chains', fontsize=8, ha='center')

# 4. Decision-Making Action
decision_box = FancyBboxPatch((7.2, y_synthesis), 2, 1.5,
                              boxstyle="round,pad=0.08",
                              edgecolor='#E64A19', linewidth=2,
                              facecolor=color_decision)
ax.add_patch(decision_box)
ax.text(8.2, y_synthesis + 1.2, '4. Decision-', fontsize=11, fontweight='bold', ha='center')
ax.text(8.2, y_synthesis + 1.0, 'Making Action', fontsize=11, fontweight='bold', ha='center')
ax.text(8.2, y_synthesis + 0.65, 'Explore action', fontsize=8, ha='center')
ax.text(8.2, y_synthesis + 0.45, 'space', fontsize=8, ha='center')
ax.text(8.2, y_synthesis + 0.25, '↓', fontsize=10, ha='center')
ax.text(8.2, y_synthesis + 0.05, 'Decision sequences', fontsize=8, ha='center')

# Arrows from memory to synthesis types
arrow_memory_q = FancyArrowPatch((2.0, 7.5), (1.3, 7.0),
                                 arrowstyle='->', mutation_scale=20,
                                 linewidth=1.5, color='#0288D1')
ax.add_patch(arrow_memory_q)

arrow_memory_p = FancyArrowPatch((3.5, 7.5), (3.6, 7.0),
                                 arrowstyle='->', mutation_scale=20,
                                 linewidth=1.5, color='#0288D1')
ax.add_patch(arrow_memory_p)

arrow_memory_r = FancyArrowPatch((6.5, 7.5), (5.9, 7.0),
                                 arrowstyle='->', mutation_scale=20,
                                 linewidth=1.5, color='#0288D1')
ax.add_patch(arrow_memory_r)

# ============ SECTION 3: Agent Workflow Cycle (Bottom Center) ============
y_cycle = 2.5

# Central cycle box
cycle_box = FancyBboxPatch((2.5, y_cycle - 0.5), 5, 2.5,
                           boxstyle="round,pad=0.1",
                           edgecolor='#F9A825', linewidth=3,
                           facecolor=color_agent_cycle, alpha=0.3)
ax.add_patch(cycle_box)

ax.text(5, y_cycle + 1.7, 'Typical Agent Workflow Cycle',
        fontsize=12, fontweight='bold', ha='center')

# Workflow components in a cycle
cycle_radius = 0.8
cycle_center_x, cycle_center_y = 5, y_cycle + 0.6

# Problem (Start)
problem_circle = Circle((cycle_center_x - 1.5, cycle_center_y), 0.35,
                        edgecolor='#D32F2F', linewidth=2, facecolor='#FFCDD2')
ax.add_patch(problem_circle)
ax.text(cycle_center_x - 1.5, cycle_center_y, 'Problem',
        fontsize=8, ha='center', va='center', fontweight='bold')

# Reflection
reflection_circle = Circle((cycle_center_x, cycle_center_y + 1.0), 0.35,
                           edgecolor='#1976D2', linewidth=2, facecolor='#BBDEFB')
ax.add_patch(reflection_circle)
ax.text(cycle_center_x, cycle_center_y + 1.0, 'Reflect',
        fontsize=8, ha='center', va='center', fontweight='bold')

# Action
action_circle = Circle((cycle_center_x + 1.5, cycle_center_y), 0.35,
                       edgecolor='#388E3C', linewidth=2, facecolor='#C8E6C9')
ax.add_patch(action_circle)
ax.text(cycle_center_x + 1.5, cycle_center_y, 'Action',
        fontsize=8, ha='center', va='center', fontweight='bold')

# Solution (End)
solution_circle = Circle((cycle_center_x, cycle_center_y - 0.8), 0.35,
                         edgecolor='#7B1FA2', linewidth=2, facecolor='#E1BEE7')
ax.add_patch(solution_circle)
ax.text(cycle_center_x, cycle_center_y - 0.8, 'Solution',
        fontsize=8, ha='center', va='center', fontweight='bold')

# Arrows showing cycle
arrow1 = FancyArrowPatch((cycle_center_x - 1.2, cycle_center_y + 0.25),
                         (cycle_center_x - 0.3, cycle_center_y + 0.8),
                         arrowstyle='->', mutation_scale=15, linewidth=2,
                         color='#424242', linestyle='--')
ax.add_patch(arrow1)

arrow2 = FancyArrowPatch((cycle_center_x + 0.3, cycle_center_y + 0.8),
                         (cycle_center_x + 1.2, cycle_center_y + 0.25),
                         arrowstyle='->', mutation_scale=15, linewidth=2,
                         color='#424242', linestyle='--')
ax.add_patch(arrow2)

arrow3 = FancyArrowPatch((cycle_center_x + 1.2, cycle_center_y - 0.25),
                         (cycle_center_x + 0.3, cycle_center_y - 0.6),
                         arrowstyle='->', mutation_scale=15, linewidth=2,
                         color='#424242', linestyle='--')
ax.add_patch(arrow3)

# Iterative loop arrow (back to reflection)
arrow_loop = FancyArrowPatch((cycle_center_x - 0.3, cycle_center_y - 0.6),
                             (cycle_center_x - 1.2, cycle_center_y - 0.25),
                             arrowstyle='->', mutation_scale=15, linewidth=2,
                             color='#F57C00', linestyle='-.')
ax.add_patch(arrow_loop)
ax.text(cycle_center_x - 1.2, cycle_center_y - 0.6, 'Iterate',
        fontsize=7, ha='center', color='#F57C00', fontweight='bold')

# Arrows connecting synthesis to cycle
arrow_q_to_cycle = FancyArrowPatch((1.3, y_synthesis), (3.65, y_cycle + 2.0),
                                   arrowstyle='->', mutation_scale=15,
                                   linewidth=1.5, color='#F57C00', linestyle=':')
ax.add_patch(arrow_q_to_cycle)

arrow_p_to_cycle = FancyArrowPatch((3.6, y_synthesis), (4.3, y_cycle + 2.0),
                                   arrowstyle='->', mutation_scale=15,
                                   linewidth=1.5, color='#0288D1', linestyle=':')
ax.add_patch(arrow_p_to_cycle)

arrow_r_to_cycle = FancyArrowPatch((5.9, y_synthesis), (5.5, y_cycle + 2.0),
                                   arrowstyle='->', mutation_scale=15,
                                   linewidth=1.5, color='#689F38', linestyle=':')
ax.add_patch(arrow_r_to_cycle)

arrow_d_to_cycle = FancyArrowPatch((8.2, y_synthesis), (6.5, y_cycle + 2.0),
                                   arrowstyle='->', mutation_scale=15,
                                   linewidth=1.5, color='#E64A19', linestyle=':')
ax.add_patch(arrow_d_to_cycle)

# ============ SECTION 4: Environment Scaling (Bottom) ============
y_env = 0.5

env_box = FancyBboxPatch((0.5, y_env), 9, 1.2,
                         boxstyle="round,pad=0.1",
                         edgecolor='#7B1FA2', linewidth=2.5,
                         facecolor=color_env)
ax.add_patch(env_box)
ax.text(5, y_env + 0.85, 'General Function-calling Data Synthesis via Environment Scaling',
        fontsize=12, fontweight='bold', ha='center')
ax.text(5, y_env + 0.45, 'Scalable framework: Heterogeneous environments as read-write databases',
        fontsize=9, ha='center')
ax.text(5, y_env + 0.15, 'Broadens function-calling scenarios → General agentic capability',
        fontsize=9, ha='center', style='italic')

# Arrow from environment to cycle
arrow_env_to_cycle = FancyArrowPatch((5, y_env + 1.2), (5, y_cycle - 0.5),
                                     arrowstyle='<->', mutation_scale=20,
                                     linewidth=2, color='#7B1FA2')
ax.add_patch(arrow_env_to_cycle)
ax.text(5.5, y_cycle - 0.2, 'Interaction', fontsize=8, color='#7B1FA2', fontweight='bold')

# ============ Add Legend/Key Points ============
ax.text(0.3, 4.7, 'Key Flow:', fontsize=10, fontweight='bold')
ax.text(0.3, 4.4, '1. Memory provides knowledge foundation', fontsize=8)
ax.text(0.3, 4.15, '2. Four synthesis types create training data', fontsize=8)
ax.text(0.3, 3.9, '3. Data captures agent workflow lifecycle', fontsize=8)
ax.text(0.3, 3.65, '4. Environment scaling enhances generalization', fontsize=8)
ax.text(0.3, 3.4, '5. All data → Mid-training phase', fontsize=8, fontweight='bold', color='#D32F2F')

# Add data flow annotations
ax.text(9.7, 7.2, 'Data', fontsize=8, ha='right', style='italic', color='gray')
ax.text(9.7, 7.0, 'Sources', fontsize=8, ha='right', style='italic', color='gray')

ax.text(9.7, 5.5, 'Training', fontsize=8, ha='right', style='italic', color='gray')
ax.text(9.7, 5.3, 'Data', fontsize=8, ha='right', style='italic', color='gray')

ax.text(9.7, 2.5, 'Agent', fontsize=8, ha='right', style='italic', color='gray')
ax.text(9.7, 2.3, 'Execution', fontsize=8, ha='right', style='italic', color='gray')

ax.text(9.7, 0.8, 'Capability', fontsize=8, ha='right', style='italic', color='gray')
ax.text(9.7, 0.6, 'Enhancement', fontsize=8, ha='right', style='italic', color='gray')

plt.tight_layout()
plt.savefig('/Volumes/ORICO/Github_Project_Remote/DeepResearch/data_synthesis_diagram.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Diagram saved to: data_synthesis_diagram.png")
plt.close()
