# GitHub Copilot Instructions

## Personality & Tone
- Be friendly, professional, and encouraging
- Use clear, concise explanations
- Show enthusiasm for data science and sports analytics

## Code Style Preferences
- Follow PEP 8 style guidelines for Python
- Use meaningful variable names (prefer descriptive over short)
- Add docstrings to functions and classes
- Include type hints where appropriate
- Prefer pandas and numpy for data manipulation
- Prefer Polars for high-performance data processing when appropriate
- Use matplotlib/seaborn for visualizations

## Project Context
- This is a deep learning project focused on football/soccer analytics
- We're following a course on deep learning applications in soccer
- This repository contains datasets, notebooks, and scripts related to the course
- Due to RAM limitations, we may need to rewrite code to use Polars instead of Pandas
- We want to migrate to polars while keeping the same functionality
- Working with tracking data, physical data, and match metadata
- Data sources include Premier League and Real Madrid datasets
- Common tasks involve data processing, visualization, and clustering analysis

## Specific Guidelines
- When working with tracking data, always consider timestamp synchronization
- Prefer Parquet format for large datasets over JSON
- Use matplotlib/seaborn for visualizations with proper labels and titles
- When suggesting ML models, explain the reasoning behind the choice
- Always handle missing data gracefully
- Include error handling in data processing code
- Always consult the original notebook for context before making changes
- When rewriting code, ensure the new code is functionally equivalent to the original
- Never make changes to the original notebook without explicit instructions

## File Organization
- Keep notebooks clean with clear markdown sections
- Don't delete any existing markdown explanations in notebooks
- Separate utility functions into .py modules when appropriate
- Use descriptive filenames that indicate content

## When Explaining Concepts
- Start with high-level overview, then dive into details
- Use football/soccer analogies when explaining technical concepts
- Provide examples relevant to sports analytics when possible
- Reference specific files in the workspace when applicable

## Libraries to Favor
- pandas, numpy for data manipulation
- matplotlib, seaborn, plotly for visualization
- scikit-learn for traditional ML
- PyTorch or TensorFlow for deep learning
- polars for high-performance data processing (when appropriate)

## What to Avoid
- Avoid overly complex solutions when simple ones suffice
- Don't use deprecated pandas methods (e.g., append, use concat instead)
- Avoid hardcoded paths; use pathlib or os.path
- Don't ignore data validation and sanity checks
