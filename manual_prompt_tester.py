#!/usr/bin/env python3
import json
import random
from sklearn.metrics import accuracy_score, classification_report

def load_data():
    """Load the dataset."""
    with open('data/train_big.json', 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples")
    return data

def sample_data(data, n=9):
    """Get random balanced sample."""
    groups = {'D': [], 'C': [], 'N': []}
    for ex in data:
        if ex['label'] in groups:
            groups[ex['label']].append(ex)
    
    each = n // 3
    sample = []
    for label, examples in groups.items():
        sample.extend(random.sample(examples, each))
    
    random.shuffle(sample)
    print(f"Sampled {len(sample)} examples ({each} per class)")
    return sample

def get_few_shot_examples(data):
    """Get few-shot examples from the dataset."""
    groups = {'D': [], 'C': [], 'N': []}
    for ex in data:
        if ex['label'] in groups:
            groups[ex['label']].append(ex)
    
    # Pick 1-2 clear examples from each category
    examples = []
    for label in ['D', 'C', 'N']:
        examples.extend(random.sample(groups[label], min(2, len(groups[label]))))
    
    return examples

def build_prompt(sample, few_shot=False, data=None):
    """Build the prompt."""
    prompt = """You are given belief pairs extracted from social media posts.

Follow this 3-step flowchart to annotate the relationship between beliefs:

- Step 1: Is the parsing and segmentation adequate to judge the relationship? If NO → Label is NEITHER
- Step 2: Are the two beliefs logically contradictory (directly or indirectly)? If YES → Label is DISSONANCE  
- Step 3: Are the two beliefs in agreement (supporting, repeating, clarifying, agreeing)? If YES → Label is CONSONANCE
- If none of the above apply → Label is NEITHER

Return a list of dicts like:
[
  {"steps": ["Yes", "Yes", "-"], "label": "D"},
  {"steps": ["Yes", "No", "Yes"], "label": "C"},
  ...
]
"""
    
    if few_shot and data:
        examples = get_few_shot_examples(data)
        prompt += "\nEXAMPLES:\n"
        
        for i, ex in enumerate(examples):
            label_map = {'D': 'DISSONANCE', 'C': 'CONSONANCE', 'N': 'NEITHER'}
            step_map = {
                'D': '["Yes", "Yes", "-"]',
                'C': '["Yes", "No", "Yes"]', 
                'N': '["No", "-", "-"]'
            }
            
            prompt += f'{i+1}. Belief 1: "{ex["du1"].strip()}" | Belief 2: "{ex["du2"].strip()}"\n'
            prompt += f'   → {{"steps": {step_map[ex["label"]]}, "label": "{ex["label"]}"}} ({label_map[ex["label"]]})\n\n'
    
    prompt += "Now classify the following:\n"
    
    for i, ex in enumerate(sample):
        prompt += f"{i+1}. Belief 1: \"{ex['du1'].strip()}\"\n   Belief 2: \"{ex['du2'].strip()}\"\n\n"
    
    return prompt

def parse_response(response, expected_len):
    """Parse LLM response."""
    try:
        parsed = json.loads(response.strip())
        if len(parsed) != expected_len:
            print(f"Expected {expected_len} items, got {len(parsed)}")
            return None
        return [item['label'].upper()[0] for item in parsed]
    except Exception as e:
        print(f"Parse error: {e}")
        return None

def evaluate(predicted, gold, sample):
    """Evaluate predictions."""
    accuracy = accuracy_score(gold, predicted)
    
    print(f"\nACCURACY: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(gold, predicted, target_names=['Consonance', 'Dissonance', 'Neither'], 
                              labels=['C', 'D', 'N'], zero_division=0))
    
    print("\nIndividual Results:")
    labels = {'C': 'Consonance', 'D': 'Dissonance', 'N': 'Neither'}
    for i, (g, p) in enumerate(zip(gold, predicted)):
        check = "✅" if g == p else "❌"
        print(f"{i+1}. Gold: {labels[g]} | Pred: {labels[p]} {check}")

def main():
    print("Cognitive Dissonance Evaluator")
    print("=" * 50)
    
    # Load and sample data
    data = load_data()
    n = int(input("Number of examples (default 9): ") or "9")
    sample = sample_data(data, n)
    
    # Choose prompt type
    mode = input("Prompt type - (z)ero-shot or (f)ew-shot? (default z): ").lower()
    few_shot = mode.startswith('f')
    
    # Build and show prompt
    prompt = build_prompt(sample, few_shot, data if few_shot else None)
    prompt_type = "FEW-SHOT" if few_shot else "ZERO-SHOT"
    print(f"\nCOPY THIS {prompt_type} PROMPT:")
    print("=" * 50)
    print(prompt)
    print("=" * 50)
    
    # Get response
    print("\nPaste LLM response:")
    response = input()
    
    # Parse and evaluate
    predicted = parse_response(response, len(sample))
    if predicted:
        gold = [ex['label'] for ex in sample]
        evaluate(predicted, gold, sample)

if __name__ == "__main__":
    main()