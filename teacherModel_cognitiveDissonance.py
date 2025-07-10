#!/usr/bin/env python3
"""
Server Teacher Model for Cognitive Dissonance Detection
Adapted for server environment with API calls
"""

import json
import pandas as pd
import requests
import argparse
import re
import random
from sklearn.metrics import accuracy_score, classification_report
import os
import time

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True, help="Model name (e.g., llama-3.1-8b-instruct)")
parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for generation")
parser.add_argument("--n_examples", type=int, default=1000, help="Number of examples to annotate")
parser.add_argument("--few_shot", type=int, default=3, help="Number of few-shot examples (0 for none)")
parser.add_argument("--data_file", type=str, default="train_big.json", help="Input data file name")
parser.add_argument("--output_suffix", type=str, default="", help="Suffix for output filename")
parser.add_argument("--balanced_sampling", action="store_true", help="Use balanced sampling for test data")
args = parser.parse_args()

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_NAME = args.model_name
TEMPERATURE = args.temperature
N_EXAMPLES = args.n_examples
FEW_SHOT = args.few_shot
DATA_FILE = args.data_file
OUTPUT_SUFFIX = args.output_suffix
BALANCED_SAMPLING = args.balanced_sampling

# Server paths
DATA_PATH = "/chronos_data/dumrania/data"
MODEL_PATH = f"/chronos_data/pretrained_models/{MODEL_NAME}"
API_URL = "http://localhost:8003/v1/completions"

# Output filename
output_filename = f"{MODEL_NAME}_cognitive_dissonance_annotations_{N_EXAMPLES}examples"
if FEW_SHOT > 0:
    output_filename += f"_{FEW_SHOT}shot"
if TEMPERATURE > 0:
    output_filename += f"_temp{TEMPERATURE}"
if BALANCED_SAMPLING:
    output_filename += f"_balanced"
if OUTPUT_SUFFIX:
    output_filename += f"_{OUTPUT_SUFFIX}"
output_filename += ".csv"

print(f"🚀 COGNITIVE DISSONANCE TEACHER MODEL - SERVER VERSION")
print(f"=" * 60)
print(f"Configuration:")
print(f"  Model: {MODEL_NAME}")
print(f"  Model path: {MODEL_PATH}")
print(f"  Temperature: {TEMPERATURE}")
print(f"  Examples to annotate: {N_EXAMPLES}")
print(f"  Few-shot examples: {FEW_SHOT}")
print(f"  Balanced sampling: {BALANCED_SAMPLING}")
print(f"  Data file: {DATA_FILE}")
print(f"  Output file: {output_filename}")
print(f"  API URL: {API_URL}")

# =============================================================================
# PROMPT TEMPLATES AND FEW-SHOT HANDLING
# =============================================================================

def format_example(example, show_reasoning=True):
    """Format a single example for few-shot prompting"""
    label_map = {'D': 'DISSONANCE', 'C': 'CONSONANCE', 'N': 'NEITHER'}
    label_name = label_map.get(example['label'], 'NEITHER')
    
    if show_reasoning:
        # Add step-by-step reasoning for chain-of-thought prompts
        step_map = {
            'D': ['YES', 'YES', 'NO'],  # Adequate parsing, contradictory, not agreeing
            'C': ['YES', 'NO', 'YES'],  # Adequate parsing, not contradictory, agreeing
            'N': ['NO', 'NO', 'NO']     # Not adequate parsing or unclear relationship
        }
        steps = step_map.get(example['label'], ['NO', 'NO', 'NO'])
        
        return f"""<post>{example['message']}</post>
<discourse>
<unit1>{example['du1']}</unit1>
<unit2>{example['du2']}</unit2>
</discourse>
<response>
<steps>
<step1>{steps[0]}</step1>
<step2>{steps[1]}</step2>
<step3>{steps[2]}</step3>
</steps>
<label>{label_name}</label>
</response>"""
    else:
        return f"""<post>{example['message']}</post>
<discourse>
<unit1>{example['du1']}</unit1>
<unit2>{example['du2']}</unit2>
</discourse>
<response>
<label>{label_name}</label>
</response>"""

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

def create_few_shot_section(examples):
    """Create the few-shot examples section for prompts"""
    if not examples:
        return ""
    
    formatted_examples = []
    for example in examples:
        formatted_examples.append(format_example(example, show_reasoning=True))
    
    section = "**Examples:**\n\n" + "\n\n".join(formatted_examples) + "\n\n**Now annotate the following:**\n\n"
    return section

def build_prompt(example, few_shot_examples=None):
    """Build the complete prompt for cognitive dissonance detection"""
    
    # Create few-shot section
    examples_section = ""
    if few_shot_examples:
        examples_section = create_few_shot_section(few_shot_examples)
    
    prompt = f"""You are an annotator for an NLP task that detects cognitive dissonance from social media. Your job is to annotate the relationship between two discourse units from Twitter posts.

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

{examples_section}

Think through each step carefully, then output:
<response>
<steps>
<step1>answer1</step1>
<step2>answer2</step2>
<step3>answer3</step3>
</steps>
<label>FINAL_LABEL</label>
</response>

<post>{example['message']}</post>
<discourse>
<unit1>{example['du1']}</unit1>
<unit2>{example['du2']}</unit2>
</discourse>"""
    
    return prompt

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """Load the cognitive dissonance dataset"""
    filepath = os.path.join(DATA_PATH, DATA_FILE)
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Loaded {len(data)} examples from {filepath}")
        
        # Show label distribution
        label_counts = {}
        for item in data:
            label = item.get('label', 'unknown')
            label_counts[label] = label_counts.get(label, 0) + 1
        print(f"   Label distribution: {label_counts}")
        
        return data
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find {filepath}")
        print(f"Available files in {DATA_PATH}:")
        try:
            files = os.listdir(DATA_PATH)
            for f in files:
                if f.endswith('.json'):
                    print(f"  - {f}")
        except:
            print("  Could not list directory")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def create_annotation_sample(data, n_examples, exclude_examples=None, balanced=False):
    """Create sample for annotation, optionally balanced"""
    available_data = data
    if exclude_examples:
        exclude_ids = {id(ex) for ex in exclude_examples}
        available_data = [ex for ex in data if id(ex) not in exclude_ids]
    
    if len(available_data) < n_examples:
        print(f"⚠️  Warning: Only {len(available_data)} examples available (requested {n_examples})")
        return available_data
    
    if balanced:
        # Create balanced sample
        groups = {'D': [], 'C': [], 'N': []}
        for ex in available_data:
            label = ex['label']
            if label in groups:
                groups[label].append(ex)
        
        sample = []
        per_group = n_examples // 3
        remainder = n_examples % 3
        
        for i, (label, examples) in enumerate([('D', groups['D']), ('C', groups['C']), ('N', groups['N'])]):
            random.shuffle(examples)
            take = per_group + (1 if i < remainder else 0)
            sample.extend(examples[:take])
        
        # Shuffle final sample
        random.shuffle(sample)
        return sample[:n_examples]
    else:
        # Random sampling
        return random.sample(available_data, n_examples)

# =============================================================================
# RESPONSE PARSING
# =============================================================================

def parse_response(response):
    """Extract classification from response with enhanced patterns"""
    response = response.strip()
    
    # Try XML parsing first
    try:
        import xml.etree.ElementTree as ET
        
        if '<response>' in response and '</response>' in response:
            start = response.find('<response>')
            end = response.find('</response>') + 11
            xml_str = response[start:end]
            
            root = ET.fromstring(xml_str)
            
            label_elem = root.find('label')
            if label_elem is not None:
                label = label_elem.text.strip().upper()
                if label in ['DISSONANCE', 'CONSONANCE', 'NEITHER']:
                    label_map = {'DISSONANCE': 'D', 'CONSONANCE': 'C', 'NEITHER': 'N'}
                    
                    steps = []
                    steps_elem = root.find('steps')
                    if steps_elem is not None:
                        for i in range(1, 4):
                            step_elem = steps_elem.find(f'step{i}')
                            if step_elem is not None:
                                steps.append(step_elem.text.strip())
                    
                    return label_map[label], steps
    except Exception:
        pass
    
    # Enhanced fallback parsing
    response_lower = response.lower()
    
    # Look for various patterns
    patterns = [
        # Direct mentions
        ('dissonance', 'D'),
        ('consonance', 'C'), 
        ('neither', 'N'),
        # With labels
        ('label: dissonance', 'D'),
        ('label: consonance', 'C'),
        ('label: neither', 'N'),
        # In context
        ('is dissonance', 'D'),
        ('shows dissonance', 'D'),
        ('indicates dissonance', 'D'),
        ('is consonance', 'C'),
        ('shows consonance', 'C'),
        ('indicates consonance', 'C'),
        ('is neither', 'N'),
        ('shows neither', 'N'),
        # Answer format
        ('answer: dissonance', 'D'),
        ('answer: consonance', 'C'),
        ('answer: neither', 'N'),
    ]
    
    for pattern, label in patterns:
        if pattern in response_lower:
            return label, []
    
    # Default
    return 'N', []

# =============================================================================
# API INTERACTION
# =============================================================================

def classify_example(example, few_shot_examples=None):
    """Send example to LLM API for classification"""
    prompt = build_prompt(example, few_shot_examples)
    
    payload = {
        "model": MODEL_PATH,
        "prompt": prompt,
        "max_tokens": 300,
        "temperature": TEMPERATURE,
        "top_p": 1.0
    }
    
    try:
        response = requests.post(API_URL, json=payload, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result["choices"][0]["text"].strip()
            
            # Parse the response
            pred_label, steps = parse_response(generated_text)
            
            return {
                "message": example["message"],
                "du1": example["du1"],
                "du2": example["du2"],
                "true_label": example["label"],
                "predicted_label": pred_label,
                "step1": steps[0] if len(steps) > 0 else "",
                "step2": steps[1] if len(steps) > 1 else "",
                "step3": steps[2] if len(steps) > 2 else "",
                "raw_response": generated_text,
                "api_status": "success"
            }
        else:
            return {
                "message": example["message"],
                "du1": example["du1"],
                "du2": example["du2"],
                "true_label": example["label"],
                "predicted_label": "ERROR",
                "step1": "",
                "step2": "",
                "step3": "",
                "raw_response": f"API Error: {response.status_code}",
                "api_status": "error"
            }
    
    except Exception as e:
        return {
            "message": example["message"],
            "du1": example["du1"],
            "du2": example["du2"],
            "true_label": example["label"],
            "predicted_label": "ERROR",
            "step1": "",
            "step2": "",
            "step3": "",
            "raw_response": f"Exception: {str(e)}",
            "api_status": "error"
        }

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    print(f"\n🔄 Starting annotation process...")
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Load data
    data = load_data()
    if not data:
        return
    
    # Select few-shot examples if requested
    few_shot_examples = None
    if FEW_SHOT > 0:
        print(f"\n📝 Selecting {FEW_SHOT} few-shot examples...")
        few_shot_examples = select_examples(data, FEW_SHOT, balanced=True)
        
        print("Selected few-shot examples:")
        for i, ex in enumerate(few_shot_examples, 1):
            print(f"  {i}. {ex['label']}: {ex['message'][:60]}...")
    
    # Create annotation sample
    print(f"\n📊 Creating annotation sample...")
    annotation_sample = create_annotation_sample(
        data, N_EXAMPLES, 
        exclude_examples=few_shot_examples, 
        balanced=BALANCED_SAMPLING
    )
    
    print(f"Will annotate {len(annotation_sample)} examples")
    
    # Label distribution
    label_counts = {}
    for ex in annotation_sample:
        label_counts[ex['label']] = label_counts.get(ex['label'], 0) + 1
    print(f"Sample label distribution: {label_counts}")
    
    # Test API connection
    print(f"\n🔍 Testing API connection...")
    test_example = annotation_sample[0]
    test_prompt = "Hello, respond with 'API working'"
    
    test_payload = {
        "model": MODEL_PATH,
        "prompt": test_prompt,
        "max_tokens": 50,
        "temperature": 0.0
    }
    
    try:
        test_response = requests.post(API_URL, json=test_payload, timeout=60)
        if test_response.status_code == 200:
            print("✅ API connection successful")
        else:
            print(f"⚠️  API returned status {test_response.status_code}")
    except Exception as e:
        print(f"❌ API connection failed: {e}")
        print("Please check if the model server is running")
        return
    
    # Process examples
    print(f"\n🔄 Processing {len(annotation_sample)} examples...")
    print(f"Progress will be shown every 25 examples...")
    
    results = []
    successful = 0
    start_time = time.time()
    
    for i, example in enumerate(annotation_sample):
        result = classify_example(example, few_shot_examples)
        results.append(result)
        
        if result["api_status"] == "success":
            successful += 1
        
        # Progress update every 25 examples
        if (i + 1) % 25 == 0 or i == len(annotation_sample) - 1:
            elapsed_time = time.time() - start_time
            accuracy = successful / (i + 1) if i + 1 > 0 else 0
            rate = (i + 1) / elapsed_time
            eta = (len(annotation_sample) - i - 1) / rate if rate > 0 else 0
            
            print(f"  Progress: {i+1:4d}/{len(annotation_sample)} | "
                  f"Success: {successful:3d}/{i+1} ({accuracy:.1%}) | "
                  f"Rate: {rate:.1f}/min | "
                  f"ETA: {eta/60:.1f}min")
    
    # Save results
    print(f"\n💾 Saving results to {output_filename}...")
    df = pd.DataFrame(results)
    df.to_csv(output_filename, index=False)
    print(f"✅ Results saved!")
    
    # Calculate metrics
    print(f"\n📊 EVALUATION METRICS")
    print("=" * 40)
    
    # Filter successful predictions only
    successful_results = df[df["api_status"] == "success"]
    total_time = time.time() - start_time
    
    print(f"Total processing time: {total_time/60:.1f} minutes")
    print(f"Average time per example: {total_time/len(annotation_sample):.1f} seconds")
    print(f"Successful predictions: {len(successful_results)}/{len(df)} ({len(successful_results)/len(df):.1%})")
    
    if len(successful_results) > 0:
        true_labels = successful_results["true_label"].tolist()
        pred_labels = successful_results["predicted_label"].tolist()
        
        accuracy = accuracy_score(true_labels, pred_labels)
        print(f"Accuracy: {accuracy:.3f} ({accuracy:.1%})")
        
        # Detailed breakdown
        print(f"\nClassification Report:")
        try:
            report = classification_report(
                true_labels, pred_labels,
                target_names=['Consonance', 'Dissonance', 'Neither'],
                labels=['C', 'D', 'N'],
                zero_division=0
            )
            print(report)
        except:
            print("Could not generate detailed report")
        
        # Confusion matrix style breakdown
        print(f"\nLabel-wise Accuracy:")
        for true_label in ['D', 'C', 'N']:
            true_subset = [i for i, label in enumerate(true_labels) if label == true_label]
            if true_subset:
                pred_subset = [pred_labels[i] for i in true_subset]
                label_accuracy = sum(1 for i, p in enumerate(pred_subset) if p == true_label) / len(pred_subset)
                label_names = {'D': 'Dissonance', 'C': 'Consonance', 'N': 'Neither'}
                print(f"  {label_names[true_label]:12s}: {label_accuracy:.3f} ({len(true_subset):3d} examples)")
        
        # Error analysis
        errors = df[(df["api_status"] == "success") & (df["true_label"] != df["predicted_label"])]
        if len(errors) > 0:
            print(f"\nError Analysis ({len(errors)} errors):")
            error_types = {}
            for _, row in errors.iterrows():
                error_type = f"{row['true_label']} → {row['predicted_label']}"
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            for error_type, count in error_types.items():
                print(f"  {error_type}: {count:3d} cases ({count/len(errors):.1%})")
    
    else:
        print("❌ No successful predictions to evaluate!")
    
    print(f"\n✅ Teacher model annotation complete!")
    print(f"Results saved to: {output_filename}")

if __name__ == "__main__":
    main()