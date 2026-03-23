# Explore the front properties in the groups table

## Goals

- Explore relationships between front properties in the groups table
- Identify groups of fronts that are similar in their properties.

## Here is how the actiity should be organized

### Code

Place any code in a fronts/dev/groups/py directory.

The primary data file is in the /mnt/tank/Oceanography/data/OGCM/LLC/Fronts/group_fronts/v1/ directory and is named front_properties_20121109T12_00_00_v1_bin_A.parquet

Place any Notebooks in the fronts/dev/groups/nb/ directory.

### Data

If you need to generate intermediate files, place them in the fronts/dev/groups/data/ directory.

### Overleaf

Place any Latex files in /home/xavier/Projects/overleaf/Front_properties/

Generate a LateX file of your plan named claude_explore.tex.

Generate LateX files describing your findings and results named claude_explore_findings.tex as you work.

Output PNG figures to the Overleaf project folder.

You may push to git as you work.  The access token is in my .bashrc profile with the name OVERLEAF.

When you add new files, update the main.tex file to include them.

## Claude code

You should use Python code exclusively.

You are allowed to run safe bash commands without prompting me.

If you need to run Python code, use the "ocean14" conda environment.

You are welcome to use multiple agents to help you with the task.  And you should spend 1 hour on the task.

Use the settings.local.json file in ./.claude to run bash commands and the like.

## Modifications

### Analysis

In the analysis, ignore these columns:
- Velocities, e.g. U_median, V_median, W_median, ug_median, vg_median, etc.
- Eta
- Okubo-Weiss

# Prompts to Claude

## Run 1

1. Digest the above instructions and provide a plan for the activity.

2. Check for changes to the instructions and update your plan accordingly and generate it as a LateX file named claude_explore.tex in the Overleaf project folder.

3. Review this doc again and update your plan accordingly.

4. Check the git status of the Overleaf repo.  You may still need to push.

5. Modify the main.tex file as per the new instructions in this doc.

6. Proceed with the plan.  As noted above, you may use multiple agents to help you with the task and spend 1 hour on the task.  You may run safe bash commnds without prompting me.

## Run 2

1. Digest the above instructions and provide a plan for the activity.  Update the claude_explore.tex file in the Overleaf project folder to reflect the new plan as a new sub-section 