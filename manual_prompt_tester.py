#!/usr/bin/env python3
"""
Simple manual tester for cognitive dissonance prompts
Clean, readable version with only essential features
Enhanced with few-shot learning capabilities
"""

import json
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import re
import random

# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

def get_prompts():
    """Define prompt templates"""
    return {
        "main": """You are an annotator for an NLP task that tries to detect cognitive dissonance from social media. Your job is to annotate consonance and dissonance from Twitter posts. The task conceptualizes consonance and dissonance as a form of discourse relation between two phrases or discourse units.

To annotate, you are given:
1. A Twitter post for context
2. Two discourse units (phrases) extracted from that post
3. You must determine the relationship between these two discourse units

Follow these steps in order:

Step 1: Is the parsing and segmentation adequate to judge the relationship?
- If NO → Label is NEITHER

Step 2: Are the two beliefs logically contradictory (directly or indirectly)?
- If YES → Label is DISSONANCE

Step 3: Are the two beliefs in agreement (supporting, repeating, clarifying, agreeing)?
- If YES → Label is CONSONANCE

If none of the above apply → Label is NEITHER

{examples}

**Post:** "{message}"
**Discourse Unit 1:** "{du1}"
**Discourse Unit 2:** "{du2}"

Answer each step with YES or NO, then provide your final label.

Your output should look like the following dictionary, without any extra text or details:
{{"steps": ["YES", "NO", "YES"], "label": "CONSONANCE"}}""",

        "cot": """You are an annotator for an NLP task that detects cognitive dissonance from social media. Your job is to annotate the relationship between two discourse units from Twitter posts.

Given:
- A Twitter post (context)
- Two discourse units (phrases) from that post

Step 1: Is the parsing and segmentation adequate to judge the relationship?
If NO → Label is NEITHER

Step 2: Are the two beliefs logically contradictory (directly or indirectly)?
If YES → Label is DISSONANCE

Step 3: Are the two beliefs in agreement (supporting, repeating, clarifying, agreeing)?
If YES → Label is CONSONANCE

If none apply → Label is NEITHER

{examples}

**Post:** "{message}"
**Discourse Unit 1:** "{du1}"
**Discourse Unit 2:** "{du2}"

Think through each step carefully, then output:
{{"steps": ["answer1", "answer2", "answer3"], "label": "FINAL_LABEL"}}""",

        "simple": """You are an annotator detecting cognitive dissonance from social media posts.

**Task:** Determine the relationship between two discourse units.

**Rules:**
- DISSONANCE: Units contradict each other
- CONSONANCE: Units support/agree with each other  
- NEITHER: Units are unrelated or unclear

{examples}

**Post:** "{message}"
**Unit 1:** "{du1}"
**Unit 2:** "{du2}"

Output format:
{{"label": "YOUR_ANSWER"}}"""
    }

# =============================================================================
# FEW-SHOT EXAMPLE GENERATION
# =============================================================================

def format_example(example, show_reasoning=False):
    """Format a single example for few-shot prompting"""
    label_map = {'D': 'DISSONANCE', 'C': 'CONSONANCE', 'N': 'NEITHER'}
    label_name = label_map.get(example['label'], 'NEITHER')
    
    if show_reasoning:
        # Add reasoning for chain-of-thought prompts
        reasoning_map = {
            'D': 'The two units present contradictory beliefs.',
            'C': 'The two units support and agree with each other.',
            'N': 'The units are not clearly related or contradictory.'
        }
        reasoning = reasoning_map.get(example['label'], 'The relationship is unclear.')
        
        return f"""**Example:**
**Post:** "{example['message']}"
**Discourse Unit 1:** "{example['du1']}"
**Discourse Unit 2:** "{example['du2']}"
**Analysis:** {reasoning}
**Answer:** {{"label": "{label_name}"}}"""
    else:
        return f"""**Example:**
**Post:** "{example['message']}"
**Discourse Unit 1:** "{example['du1']}"
**Discourse Unit 2:** "{example['du2']}"
**Answer:** {{"label": "{label_name}"}}"""

def select_examples(data, n_examples=3, balanced=True):
    """Select examples for few-shot prompting"""
    if not data or n_examples == 0:
        return []
    
    if balanced:
        # Try to get balanced examples across labels
        groups = {'D': [], 'C': [], 'N': []}
        for example in data:
            label = example['label']
            if label in groups:
                groups[label].append(example)
        
        # Shuffle each group
        for group in groups.values():
            random.shuffle(group)
        
        # Take examples from each group
        examples = []
        per_group = max(1, n_examples // 3)
        
        for label in ['D', 'C', 'N']:  # Ensure consistent order
            examples.extend(groups[label][:per_group])
        
        # If we need more examples, add randomly
        if len(examples) < n_examples:
            remaining = [ex for ex in data if ex not in examples]
            random.shuffle(remaining)
            examples.extend(remaining[:n_examples - len(examples)])
        
        return examples[:n_examples]
    else:
        # Random selection
        selected = random.sample(data, min(n_examples, len(data)))
        return selected

def create_few_shot_section(examples, prompt_type="simple"):
    """Create the few-shot examples section for prompts"""
    if not examples:
        return ""
    
    show_reasoning = prompt_type == "cot"
    
    formatted_examples = []
    for example in examples:
        formatted_examples.append(format_example(example, show_reasoning))
    
    section = "**Examples:**\n\n" + "\n\n".join(formatted_examples) + "\n\n**Now annotate the following:**\n"
    return section

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """Load the dissonance dataset"""
    print("Loading data...")
    
    # Try to find dataset files
    files = ['data/train_big.json', 'train_big.json', 'data/dev.json', 'dev.json']
    
    for filepath in files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Loaded {len(data)} examples from {filepath}")
            return data
        except FileNotFoundError:
            continue
    
    print("No dataset files found!")
    print("Make sure you have the dissonance dataset in your directory")
    return None

def create_test_sample(data, n=10, exclude_examples=None):
    """Create a balanced test sample, excluding few-shot examples"""
    if not data:
        return None
    
    # Exclude few-shot examples from test set
    available_data = data
    if exclude_examples:
        exclude_ids = {id(ex) for ex in exclude_examples}
        available_data = [ex for ex in data if id(ex) not in exclude_ids]
    
    # Group by label
    groups = {'D': [], 'C': [], 'N': []}
    for example in available_data:
        label = example['label']
        if label in groups:
            groups[label].append(example)
    
    # Take samples from each group
    sample = []
    per_group = n // 3
    
    for label, examples in groups.items():
        random.shuffle(examples)  # Add randomness
        sample.extend(examples[:per_group])
    
    # If we need more examples, add randomly from remaining
    if len(sample) < n:
        remaining = [ex for ex in available_data if ex not in sample]
        random.shuffle(remaining)
        sample.extend(remaining[:n - len(sample)])
    
    print(f"Created test sample with {len(sample)} examples")
    return sample

# =============================================================================
# RESPONSE PARSING
# =============================================================================

def parse_response(response):
    """Extract classification from JSON-formatted LLM response"""
    response = response.strip()
    
    # Try to parse JSON response
    try:
        import json as json_lib
        # Handle cases where response might have extra text
        if '{' in response and '}' in response:
            start = response.find('{')
            end = response.rfind('}') + 1
            json_str = response[start:end]
            parsed = json_lib.loads(json_str)
            
            # Extract label
            label = parsed.get('label', '').upper()
            if label in ['DISSONANCE', 'CONSONANCE', 'NEITHER']:
                # Convert to single letter for consistency
                label_map = {'DISSONANCE': 'D', 'CONSONANCE': 'C', 'NEITHER': 'N'}
                return label_map[label], parsed.get('steps', [])
    except:
        pass
    
    # Fallback parsing if JSON fails
    response_lower = response.lower()
    if 'dissonance' in response_lower:
        return 'D', []
    elif 'consonance' in response_lower:
        return 'C', []
    elif 'neither' in response_lower:
        return 'N', []
    
    return 'N', []  # Default to Neither

# =============================================================================
# TESTING FUNCTIONS
# =============================================================================

def test_prompt(prompt_name, test_data, few_shot_examples=None):
    """Test a prompt on the data with optional few-shot examples"""
    prompts = get_prompts()
    template = prompts[prompt_name]
    
    print(f"\nTesting: {prompt_name}")
    if few_shot_examples:
        print(f"Using {len(few_shot_examples)} few-shot examples")
    print("=" * 50)
    
    # Create few-shot section
    examples_section = ""
    if few_shot_examples:
        examples_section = create_few_shot_section(few_shot_examples, prompt_name)
    
    predictions = []
    true_labels = []
    step_responses = []
    
    for i, example in enumerate(test_data):
        print(f"\n--- Example {i+1}/{len(test_data)} ---")
        print(f"True answer: {example['label']}")
        print(f"Message: {example['message'][:80]}...")
        print(f"Unit 1: {example['du1']}")
        print(f"Unit 2: {example['du2']}")
        
        # Format prompt with few-shot examples
        prompt = template.format(
            message=example['message'],
            du1=example['du1'],
            du2=example['du2'],
            examples=examples_section
        )
        
        print(f"\nCOPY THIS TO YOUR LLM:")
        print("-" * 30)
        print(prompt)
        print("-" * 30)
        
        # Get response
        response = input("\nPaste LLM response: ").strip()
        if not response:
            print("Skipping empty response")
            continue
        
        # Parse response
        pred, steps = parse_response(response)
        correct = "CORRECT" if pred == example['label'] else "INCORRECT"
        
        print(f"Result: {pred} ({correct})")
        if steps:
            print(f"Steps: {steps}")
        
        # Store results
        predictions.append(pred)
        true_labels.append(example['label'])
        step_responses.append(steps)
        
        # Continue?
        if i < len(test_data) - 1:
            cont = input("Continue? (Enter/n): ").strip()
            if cont.lower() == 'n':
                break
    
    return predictions, true_labels, step_responses

def show_results(predictions, true_labels, step_responses, prompt_name):
    """Display test results and error analysis"""
    if not predictions:
        print("No results to show!")
        return
    
    print(f"\nRESULTS: {prompt_name}")
    print("=" * 50)
    
    # Overall accuracy
    accuracy = accuracy_score(true_labels, predictions)
    print(f"Accuracy: {accuracy:.1%}")
    
    # Individual results with error analysis
    label_names = {'D': 'Dissonance', 'C': 'Consonance', 'N': 'Neither'}
    print(f"\nIndividual Results:")
    errors = []
    
    for i, (true, pred, steps) in enumerate(zip(true_labels, predictions, step_responses)):
        status = "CORRECT" if true == pred else "INCORRECT"
        print(f"  {i+1}: {label_names[true]} -> {label_names[pred]} ({status})")
        if steps:
            print(f"      Steps: {steps}")
        
        # Track errors for analysis
        if true != pred:
            errors.append({
                'example': i+1,
                'true': true,
                'predicted': pred,
                'steps': steps
            })
    
    # Error analysis
    if errors:
        print(f"\nERROR ANALYSIS:")
        print(f"Found {len(errors)} errors. Common patterns:")
        
        # Count error types
        error_types = {}
        for error in errors:
            error_type = f"{error['true']} -> {error['predicted']}"
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        for error_type, count in error_types.items():
            print(f"  {error_type}: {count} cases")
    
    # Detailed breakdown
    print(f"\nDetailed Classification Report:")
    try:
        report = classification_report(
            true_labels, predictions, 
            target_names=['Consonance', 'Dissonance', 'Neither'],
            labels=['C', 'D', 'N'],
            zero_division=0
        )
        print(report)
    except:
        print("Could not generate detailed report")
    
    return accuracy

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main testing workflow"""
    print("COGNITIVE DISSONANCE PROMPT TESTER")
    print("=" * 50)
    
    # Load data
    data = load_data()
    if not data:
        return
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Ask about few-shot examples
    use_few_shot = input("Use few-shot examples? (y/n): ").strip().lower() == 'y'
    few_shot_examples = None
    
    if use_few_shot:
        n_shots = input("How many few-shot examples? (1-10 recommended): ").strip()
        try:
            n_shots = int(n_shots)
            n_shots = max(1, min(10, n_shots))  # Limit between 1-10
        except:
            n_shots = 3
        
        balanced_shots = input("Use balanced examples across labels? (y/n): ").strip().lower() == 'y'
        few_shot_examples = select_examples(data, n_shots, balanced_shots)
        
        print(f"\nSelected {len(few_shot_examples)} few-shot examples:")
        for i, ex in enumerate(few_shot_examples, 1):
            print(f"  {i}. {ex['label']}: {ex['message'][:50]}...")
    
    # Create test sample
    n_examples = input("How many examples to test? (5-20 recommended): ").strip()
    try:
        n_examples = int(n_examples)
    except:
        n_examples = 10
    
    test_data = create_test_sample(data, n_examples, few_shot_examples)
    if not test_data:
        return
    
    # Choose prompt
    prompts = get_prompts()
    print(f"\nAvailable prompts:")
    for i, name in enumerate(prompts.keys(), 1):
        print(f"  {i}. {name}")
    
    choice = input(f"Choose prompt (1-{len(prompts)}): ").strip()
    try:
        prompt_name = list(prompts.keys())[int(choice) - 1]
    except:
        prompt_name = 'simple'
    
    # Run test
    print(f"\nTesting '{prompt_name}' on {len(test_data)} examples")
    if few_shot_examples:
        print(f"With {len(few_shot_examples)} few-shot examples")
    print("Goal: Achieve 80% accuracy")
    
    predictions, true_labels, step_responses = test_prompt(prompt_name, test_data, few_shot_examples)
    
    # Show results with error analysis
    accuracy = show_results(predictions, true_labels, step_responses, prompt_name)
    
    # Test another?
    again = input(f"\nTest another prompt? (y/n): ").strip().lower()
    if again == 'y':
        main()

if __name__ == "__main__":
    main()