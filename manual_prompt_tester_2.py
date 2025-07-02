#!/usr/bin/env python3
"""
Manual tester for cognitive dissonance prompts
Adapted from the original testing script with self-belief prompt structure
"""

import json
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import re

# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

def get_prompts():
    """Define prompt templates for cognitive dissonance detection"""
    return {
        "main": """You are an annotator for an NLP task that tries to detect cognitive dissonance from social media. Your job is to annotate consonance and dissonance from Twitter posts. The task conceptualizes consonance and dissonance as a form of discourse relation between two phrases or discourse units.

### **Class Definitions**:
- **DISSONANCE:** The two beliefs are logically contradictory (directly or indirectly).
- **CONSONANCE:** The two beliefs are in agreement (supporting, repeating, clarifying, agreeing).
- **NEITHER:** Parsing and segmentation inadequate to judge the relationship, or beliefs are unrelated.

---
### **Post to Analyze**:
{message}

### **Discourse Unit 1**:
{du1}

### **Discourse Unit 2**:
{du2}
---

Parsing Adequate: [YES/NO] | Reason: (Is the parsing and segmentation adequate to judge the relationship?)  
Beliefs Contradict: [YES/NO] | Reason: (Are the two beliefs logically contradictory directly or indirectly?)  
Beliefs Agree: [YES/NO] | Reason: (Are the two beliefs in agreement, supporting, repeating, clarifying, or agreeing?)  
Units Related: [YES/NO] | Reason: (Do the discourse units form a meaningful relationship?)
Classification: [DISSONANCE/CONSONANCE/NEITHER] → (Choose the exact word)  
Confidence: [0-100] → (Choose a confidence score from 0-100 just give the number)  

---
### **Strict Answer Format Example**: 
Parsing Adequate: YES | Reason: The parsing and segmentation are clear and adequate.  
Beliefs Contradict: NO | Reason: No logical contradiction between the two discourse units.  
Beliefs Agree: YES | Reason: The second unit supports and clarifies the first unit.  
Units Related: YES | Reason: Both units discuss related aspects of the same topic.
CONSONANCE  
85  

---
### **Decision Steps** (Answer ALL questions, then determine label):
**Step 1:** Is the parsing and segmentation adequate to judge the relationship?
- Always answer YES or NO

**Step 2:** Are the two beliefs logically contradictory (directly or indirectly)?
- Always answer YES or NO
- If YES → Classification should be DISSONANCE

**Step 3:** Are the two beliefs in agreement (supporting, repeating, clarifying, agreeing)?
- Always answer YES or NO  
- If YES → Classification should be CONSONANCE

**Classification Logic:**
- If Step 1 = NO → NEITHER
- If Step 2 = YES → DISSONANCE
- If Step 3 = YES → CONSONANCE
- Otherwise → NEITHER

### **Rules**:
- **ONLY output six lines** in the exact format.
- **Answer ALL questions with YES or NO** - never skip any step.
- **Use exact words: DISSONANCE, CONSONANCE, or NEITHER** for classification.
- **DO NOT include placeholders** in the final output.
- **DO NOT repeat questions or add extra text.**

Now generate the response:""",

        "classification_first": """You are an annotator for an NLP task that tries to detect cognitive dissonance from social media. Your job is to annotate consonance and dissonance from Twitter posts. The task conceptualizes consonance and dissonance as a form of discourse relation between two phrases or discourse units.

### **Class Definitions**:
- **DISSONANCE:** The two beliefs are logically contradictory (directly or indirectly).
- **CONSONANCE:** The two beliefs are in agreement (supporting, repeating, clarifying, agreeing).
- **NEITHER:** Parsing and segmentation inadequate to judge the relationship, or beliefs are unrelated.

---
### **Post to Analyze**:
{message}

### **Discourse Unit 1**:
{du1}

### **Discourse Unit 2**:
{du2}
---

Classification: [DISSONANCE/CONSONANCE/NEITHER] → (Choose the exact word)  
Confidence: [0-100] → (Choose a confidence score from 0-100 just give the number)  
Parsing Adequate: [YES/NO] | Reason: (Is the parsing and segmentation adequate to judge the relationship?)  
Beliefs Contradict: [YES/NO] | Reason: (Are the two beliefs logically contradictory directly or indirectly?)  
Beliefs Agree: [YES/NO] | Reason: (Are the two beliefs in agreement, supporting, repeating, clarifying, or agreeing?)  
Units Related: [YES/NO] | Reason: (Do the discourse units form a meaningful relationship?)

---
### **Rules**:
- **ONLY output six lines** in the exact format.
- **Answer ALL questions with YES or NO** - never skip any step.
- **Use exact words: DISSONANCE, CONSONANCE, or NEITHER** for classification.

Now generate the response:""",

        "simple": """You are an annotator detecting cognitive dissonance from social media posts.

**Task:** Determine the relationship between two discourse units.

**Post:** {message}
**Unit 1:** {du1}
**Unit 2:** {du2}

**Rules:**
- DISSONANCE: Units contradict each other
- CONSONANCE: Units support/agree with each other  
- NEITHER: Units are unrelated or unclear

Answer ALL steps:
Parsing Adequate: [YES/NO] | Reason: (Brief reason)
Beliefs Contradict: [YES/NO] | Reason: (Brief reason)  
Beliefs Agree: [YES/NO] | Reason: (Brief reason)
Units Related: [YES/NO] | Reason: (Brief reason)
Classification: [DISSONANCE/CONSONANCE/NEITHER]
Confidence: [0-100]

Output format - exactly 6 lines:
Parsing Adequate: YES | Reason: Clear units.
Beliefs Contradict: NO | Reason: No contradiction.
Beliefs Agree: YES | Reason: Units support each other.
Units Related: YES | Reason: Same topic.
CONSONANCE
85"""
    }

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

def create_test_sample(data, n=10):
    """Create a balanced test sample"""
    if not data:
        return None
    
    # Group by label
    groups = {'D': [], 'C': [], 'N': []}
    for example in data:
        label = example['label']
        if label in groups:
            groups[label].append(example)
    
    # Take samples from each group
    sample = []
    per_group = n // 3
    
    for label, examples in groups.items():
        sample.extend(examples[:per_group])
    
    print(f"Created test sample with {len(sample)} examples")
    return sample

# =============================================================================
# RESPONSE PARSING
# =============================================================================

def parse_response(response):
    """Extract classification and steps from structured LLM response (handles both multi-line and single-line formats)"""
    response = response.strip()
    
    # Initialize results
    steps = []
    label = 'N'
    confidence = 0
    
    try:
        # Parse responses (works for both single-line and multi-line)
        parsing_adequate = "NO"
        beliefs_contradict = "NO" 
        beliefs_agree = "NO"
        units_related = "NO"
        
        # Extract Parsing Adequate
        if "Parsing Adequate:" in response:
            parsing_section = response.split("Parsing Adequate:")[1].split("Beliefs Contradict:")[0]
            parsing_adequate = "YES" if "YES" in parsing_section.upper() else "NO"
        
        # Extract Beliefs Contradict
        if "Beliefs Contradict:" in response:
            contradict_section = response.split("Beliefs Contradict:")[1].split("Beliefs Agree:")[0]
            beliefs_contradict = "YES" if "YES" in contradict_section.upper() else "NO"
        
        # Extract Beliefs Agree
        if "Beliefs Agree:" in response:
            agree_section = response.split("Beliefs Agree:")[1].split("Units Related:")[0]
            beliefs_agree = "YES" if "YES" in agree_section.upper() else "NO"
        
        # Extract Units Related
        if "Units Related:" in response:
            # Find the end of Units Related section (before DISSONANCE/CONSONANCE/NEITHER or numbers)
            related_start = response.find("Units Related:") + len("Units Related:")
            related_section = response[related_start:]
            # Stop at classification words or standalone numbers
            for word in ["DISSONANCE", "CONSONANCE", "NEITHER"]:
                if word in related_section.upper():
                    related_section = related_section.split(word)[0]
                    break
            units_related = "YES" if "YES" in related_section.upper() else "NO"
        
        # Extract classification label
        if "DISSONANCE" in response.upper():
            label = 'D'
        elif "CONSONANCE" in response.upper():
            label = 'C'
        elif "NEITHER" in response.upper():
            label = 'N'
        
        # Extract confidence (look for standalone numbers)
        import re
        # Find numbers that aren't part of other text (like "18" in the example)
        # Look for numbers at the end or after classification words
        confidence_match = None
        for word in ["DISSONANCE", "CONSONANCE", "NEITHER"]:
            if word in response.upper():
                after_classification = response.upper().split(word)[1]
                confidence_match = re.search(r'\b(\d{1,3})\b', after_classification)
                if confidence_match:
                    potential_conf = int(confidence_match.group(1))
                    if 0 <= potential_conf <= 100:  # Valid confidence range
                        confidence = potential_conf
                        break
        
        if confidence == 0:  # Fallback: look for last number in reasonable range
            numbers = re.findall(r'\b(\d{1,3})\b', response)
            for num in reversed(numbers):
                if 0 <= int(num) <= 100:
                    confidence = int(num)
                    break
        
        steps = [parsing_adequate, beliefs_contradict, beliefs_agree]
        
    except Exception as e:
        print(f"Parsing error: {e}")
        # Fallback parsing
        response_lower = response.lower()
        if 'dissonance' in response_lower:
            label = 'D'
        elif 'consonance' in response_lower:
            label = 'C'
        else:
            label = 'N'
        
        # Try to get YES/NO from the text
        steps = [
            "YES" if "parsing adequate: yes" in response_lower else "NO",
            "YES" if "beliefs contradict: yes" in response_lower else "NO", 
            "YES" if "beliefs agree: yes" in response_lower else "NO"
        ]
    
    return label, steps, confidence

# =============================================================================
# TESTING FUNCTIONS
# =============================================================================

def test_prompt(prompt_name, test_data):
    """Test a prompt on the data"""
    prompts = get_prompts()
    template = prompts[prompt_name]
    
    print(f"\nTesting: {prompt_name}")
    print("=" * 50)
    
    predictions = []
    true_labels = []
    step_responses = []
    confidences = []
    
    for i, example in enumerate(test_data):
        print(f"\n--- Example {i+1}/{len(test_data)} ---")
        print(f"True answer: {example['label']}")
        print(f"Message: {example['message'][:80]}...")
        print(f"Unit 1: {example['du1']}")
        print(f"Unit 2: {example['du2']}")
        
        # Format prompt
        prompt = template.format(
            message=example['message'],
            du1=example['du1'],
            du2=example['du2']
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
        pred, steps, confidence = parse_response(response)
        correct = "CORRECT" if pred == example['label'] else "INCORRECT"
        
        print(f"Result: {pred} ({correct})")
        print(f"Steps: {steps}")
        print(f"Confidence: {confidence}")
        
        # Store results
        predictions.append(pred)
        true_labels.append(example['label'])
        step_responses.append(steps)
        confidences.append(confidence)
        
        # Continue?
        if i < len(test_data) - 1:
            cont = input("Continue? (Enter/n): ").strip()
            if cont.lower() == 'n':
                break
    
    return predictions, true_labels, step_responses, confidences

def show_results(predictions, true_labels, step_responses, confidences, prompt_name):
    """Display test results and error analysis"""
    if not predictions:
        print("No results to show!")
        return
    
    print(f"\nRESULTS: {prompt_name}")
    print("=" * 50)
    
    # Overall accuracy
    accuracy = accuracy_score(true_labels, predictions)
    print(f"Accuracy: {accuracy:.1%}")
    
    # Average confidence
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    print(f"Average Confidence: {avg_confidence:.1f}")
    
    # Individual results with error analysis
    label_names = {'D': 'Dissonance', 'C': 'Consonance', 'N': 'Neither'}
    print(f"\nIndividual Results:")
    errors = []
    
    for i, (true, pred, steps, conf) in enumerate(zip(true_labels, predictions, step_responses, confidences)):
        status = "CORRECT" if true == pred else "INCORRECT"
        print(f"  {i+1}: {label_names[true]} -> {label_names[pred]} ({status}) [Conf: {conf}]")
        if steps:
            print(f"      Steps: Parse={steps[0]}, Contradict={steps[1]}, Agree={steps[2]}")
        
        # Track errors for analysis
        if true != pred:
            errors.append({
                'example': i+1,
                'true': true,
                'predicted': pred,
                'steps': steps,
                'confidence': conf
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
        
        # Step consistency analysis
        print(f"\nStep Consistency Check:")
        for error in errors:
            steps = error['steps']
            if len(steps) >= 3:
                parse_ok = steps[0] == "YES"
                contradict = steps[1] == "YES" 
                agree = steps[2] == "YES"
                
                expected_label = 'N'  # Default
                if parse_ok:
                    if contradict:
                        expected_label = 'D'
                    elif agree:
                        expected_label = 'C'
                
                if expected_label != error['predicted']:
                    print(f"    Example {error['example']}: Steps suggest {expected_label} but predicted {error['predicted']}")
        
        print(f"\nSUGGESTIONS:")
        if 'D -> C' in error_types or 'D -> N' in error_types:
            print("  - Model struggling to detect contradictions")
            print("  - Consider adding clearer contradiction examples")
        if 'C -> N' in error_types:
            print("  - Model not recognizing support relationships")
            print("  - Add more obvious agreement examples")
        if 'N -> D' in error_types or 'N -> C' in error_types:
            print("  - Model over-interpreting neutral cases")
            print("  - Emphasize 'unrelated' criteria in prompt")
    
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
    print("COGNITIVE DISSONANCE PROMPT TESTER (Self-Belief Format)")
    print("=" * 60)
    
    # Load data
    data = load_data()
    if not data:
        return
    
    # Create test sample
    n_examples = input("How many examples to test? (5-20 recommended): ").strip()
    try:
        n_examples = int(n_examples)
    except:
        n_examples = 10
    
    test_data = create_test_sample(data, n_examples)
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
        prompt_name = 'main'
    
    # Run test
    print(f"\nTesting '{prompt_name}' on {len(test_data)} examples")
    print("Goal: Achieve 80% accuracy with consistent step reasoning")
    predictions, true_labels, step_responses, confidences = test_prompt(prompt_name, test_data)
    
    # Show results with error analysis
    show_results(predictions, true_labels, step_responses, confidences, prompt_name)
    
    # Test another?
    again = input(f"\nTest another prompt? (y/n): ").strip().lower()
    if again == 'y':
        main()

if __name__ == "__main__":
    main()