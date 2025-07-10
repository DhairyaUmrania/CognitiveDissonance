#!/usr/bin/env python3
"""
Local direct model test for cognitive dissonance detection
Loads and runs the model directly without API calls
"""

import json
import pandas as pd
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, classification_report
import time
import os

# =============================================================================
# CONFIGURATION - MODIFY THESE FOR YOUR LOCAL SETUP
# =============================================================================

# Local model configuration
MODEL_PATH = r"C:\root\SBU\2.Spring25\CSE538\Assignment4_Project\SmartFinance\Finetuned_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 512
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.0
N_EXAMPLES = 10  # Small number for testing
FEW_SHOT = 3

print(f"🧪 LOCAL DIRECT MODEL TEST CONFIGURATION:")
print(f"  Model: {MODEL_PATH}")
print(f"  Device: {DEVICE}")
print(f"  Temperature: {TEMPERATURE}")
print(f"  Test examples: {N_EXAMPLES}")
print(f"  Few-shot examples: {FEW_SHOT}")

# Data loading configuration
DATA_PATH = "."  # Current directory, change if your data is elsewhere
DATA_FILES = ['train_big.json', 'data/train_big.json', 'dev.json', 'data/dev.json']  # Try these files in order

# =============================================================================
# DATA LOADING
# =============================================================================

def load_real_data():
    """Load the actual cognitive dissonance dataset"""
    print(f"🔍 Looking for dataset files...")
    
    for filepath in DATA_FILES:
        try:
            print(f"  Trying: {filepath}")
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ Loaded {len(data)} examples from {filepath}")
            
            # Show data structure
            if data:
                print(f"   Sample entry keys: {list(data[0].keys())}")
                
                # Show label distribution
                label_counts = {}
                for item in data:
                    label = item.get('label', 'unknown')
                    label_counts[label] = label_counts.get(label, 0) + 1
                print(f"   Label distribution: {label_counts}")
            
            return data
            
        except FileNotFoundError:
            print(f"  ❌ Not found: {filepath}")
            continue
        except Exception as e:
            print(f"  ❌ Error loading {filepath}: {e}")
            continue
    
    print("❌ No dataset files found!")
    print("💡 Available files in current directory:")
    try:
        files = [f for f in os.listdir('.') if f.endswith('.json')]
        for f in files:
            print(f"   - {f}")
    except:
        print("   Could not list directory")
    
    return None

def load_model():
    """Load the model and tokenizer using the same pattern as finetuned_llm.py"""
    print(f"🔄 Loading finetuned model from {MODEL_PATH}...")
    
    try:
        # Load model and tokenizer (same as your finetuned_llm.py)
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        
        print("Loading seq2seq model...")
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
        
        # Move to GPU if available (same logic as your file)
        device_id = 0 if torch.cuda.is_available() and DEVICE.startswith("cuda") else -1
        if device_id >= 0:
            device_name = f"cuda:{device_id}"
            model.to(device_name)
            print(f"✅ Model loaded on GPU: {device_name}")
        else:
            device_name = "cpu"
            model = model.to(device_name)
            print(f"✅ Model loaded on CPU")
        
        model.eval()
        
        print(f"✅ Model loaded successfully!")
        print(f"   Model type: Seq2Seq (T5-style)")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
        print(f"   Vocab size: {len(tokenizer)}")
        print(f"   Device: {device_name}")
        
        return model, tokenizer, device_name
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None, None

# =============================================================================
# PROMPT BUILDING (Same as before)
# =============================================================================

def format_example(example, show_reasoning=True):
    """Format a single example for few-shot prompting"""
    label_map = {'D': 'DISSONANCE', 'C': 'CONSONANCE', 'N': 'NEITHER'}
    label_name = label_map.get(example['label'], 'NEITHER')
    
    if show_reasoning:
        step_map = {
            'D': ['YES', 'YES', 'NO'],
            'C': ['YES', 'NO', 'YES'],
            'N': ['NO', 'NO', 'NO']
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

def create_few_shot_section(examples):
    """Create the few-shot examples section"""
    if not examples:
        return ""
    
    formatted_examples = []
    for example in examples:
        formatted_examples.append(format_example(example, show_reasoning=True))
    
    section = "**Examples:**\n\n" + "\n\n".join(formatted_examples) + "\n\n**Now annotate the following:**\n\n"
    return section

def build_prompt(example, few_shot_examples=None):
    """Build the complete prompt"""
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
# MODEL INFERENCE
# =============================================================================

def generate_response(model, tokenizer, prompt, device_name):
    """Generate response from the seq2seq model"""
    try:
        # Tokenize input for seq2seq model
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=MAX_LENGTH, 
            truncation=True,
            padding=True
        )
        
        # Move inputs to device
        if device_name != "cpu":
            inputs = {k: v.to(device_name) for k, v in inputs.items()}
        
        # Generate response using seq2seq model
        with torch.no_grad():
            if TEMPERATURE == 0.0:
                # Greedy decoding
                outputs = model.generate(
                    **inputs,
                    max_length=MAX_NEW_TOKENS,
                    do_sample=False,
                    num_beams=1,
                    early_stopping=True
                )
            else:
                # Sampling
                outputs = model.generate(
                    **inputs,
                    max_length=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=0.9,
                    early_stopping=True
                )
        
        # Decode response (for seq2seq, we decode the full output)
        generated_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_response
        
    except Exception as e:
        return f"GENERATION_ERROR: {str(e)}"

# =============================================================================
# RESPONSE PARSING (Same as before)
# =============================================================================

def parse_response(response):
    """Extract classification from response"""
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
    except Exception as e:
        print(f"  XML parsing failed: {e}")
    
    # Fallback parsing
    response_lower = response.lower()
    if 'dissonance' in response_lower:
        return 'D', []
    elif 'consonance' in response_lower:
        return 'C', []
    elif 'neither' in response_lower:
        return 'N', []
    
    return 'N', []

# =============================================================================
# MAIN TEST FUNCTION
# =============================================================================

def test_direct_model():
    """Test the model directly"""
    print("\n🚀 Starting Direct Model Test")
    print("=" * 50)
    
    # Load model
    model, tokenizer, device_name = load_model()
    if model is None or tokenizer is None:
        print("❌ Failed to load model. Exiting.")
        return False
    
    # Set random seed
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Load real data
    data = load_real_data()
    if data is None:
        print("❌ Failed to load dataset. Exiting.")
        return False
    
    # Select few-shot examples
    few_shot_examples = None
    if FEW_SHOT > 0:
        # Select balanced examples across all labels
        groups = {'D': [], 'C': [], 'N': []}
        for example in data:
            label = example['label']
            if label in groups:
                groups[label].append(example)
        
        # Shuffle and select
        few_shot_examples = []
        per_group = max(1, FEW_SHOT // 3)
        
        for label in ['D', 'C', 'N']:
            if groups[label]:
                random.shuffle(groups[label])
                few_shot_examples.extend(groups[label][:per_group])
        
        # Add more if needed
        if len(few_shot_examples) < FEW_SHOT:
            remaining = [ex for ex in data if ex not in few_shot_examples]
            random.shuffle(remaining)
            few_shot_examples.extend(remaining[:FEW_SHOT - len(few_shot_examples)])
        
        few_shot_examples = few_shot_examples[:FEW_SHOT]
        
        print(f"📝 Using {len(few_shot_examples)} few-shot examples:")
        for i, ex in enumerate(few_shot_examples, 1):
            print(f"  {i}. {ex['label']}: {ex['message'][:50]}...")
    
    # Create test sample (exclude few-shot examples)
    available_data = data
    if few_shot_examples:
        exclude_ids = {id(ex) for ex in few_shot_examples}
        available_data = [ex for ex in data if id(ex) not in exclude_ids]
    
    # Sample for testing, ensuring balanced representation
    if len(available_data) < N_EXAMPLES:
        print(f"⚠️  Warning: Only {len(available_data)} examples available (requested {N_EXAMPLES})")
        test_sample = available_data
    else:
        # Try to get balanced test sample
        test_groups = {'D': [], 'C': [], 'N': []}
        for ex in available_data:
            label = ex['label']
            if label in test_groups:
                test_groups[label].append(ex)
        
        test_sample = []
        per_group = N_EXAMPLES // 3
        
        for label in ['D', 'C', 'N']:
            if test_groups[label]:
                random.shuffle(test_groups[label])
                test_sample.extend(test_groups[label][:per_group])
        
        # Add more if needed
        if len(test_sample) < N_EXAMPLES:
            remaining = [ex for ex in available_data if ex not in test_sample]
            random.shuffle(remaining)
            test_sample.extend(remaining[:N_EXAMPLES - len(test_sample)])
        
        # Final shuffle
        random.shuffle(test_sample)
    
    print(f"\n🧪 Testing {len(test_sample)} examples...")
    
    # Show test sample distribution
    test_counts = {}
    for ex in test_sample:
        label = ex['label']
        test_counts[label] = test_counts.get(label, 0) + 1
    print(f"   Test distribution: {test_counts}")
    
    # Process examples
    results = []
    successful = 0
    total_time = 0
    
    for i, example in enumerate(test_sample):
        print(f"\n--- Example {i+1}/{len(test_sample)} ---")
        print(f"Message: {example['message']}")
        print(f"Unit 1: {example['du1']}")
        print(f"Unit 2: {example['du2']}")
        print(f"True label: {example['label']}")
        
        # Build prompt
        prompt = build_prompt(example, few_shot_examples)
        
        print(f"\n🔍 PROMPT FOR EXAMPLE {i+1}:")
        print("=" * 60)
        print(prompt)
        print("=" * 60)
        
        # Generate response
        print("🔄 Generating response...")
        start_time = time.time()
        response = generate_response(model, tokenizer, prompt, device_name)
        elapsed = time.time() - start_time
        total_time += elapsed
        
        if response.startswith("GENERATION_ERROR"):
            print(f"❌ Error: {response}")
            result = {
                "message": example["message"],
                "du1": example["du1"],
                "du2": example["du2"],
                "true_label": example["label"],
                "predicted_label": "ERROR",
                "step1": "",
                "step2": "",
                "step3": "",
                "raw_response": response,
                "response_time": elapsed
            }
        else:
            # Parse response
            pred_label, steps = parse_response(response)
            is_correct = pred_label == example["label"]
            if is_correct:
                successful += 1
            
            status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
            print(f"Predicted: {pred_label} ({status})")
            print(f"Response time: {elapsed:.1f}s")
            
            if steps:
                print(f"Steps: {steps}")
            
            # Show a snippet of the raw response
            response_snippet = response[:100] + "..." if len(response) > 100 else response
            print(f"Raw response: {response_snippet}")
            
            result = {
                "message": example["message"],
                "du1": example["du1"],
                "du2": example["du2"],
                "true_label": example["label"],
                "predicted_label": pred_label,
                "step1": steps[0] if len(steps) > 0 else "",
                "step2": steps[1] if len(steps) > 1 else "",
                "step3": steps[2] if len(steps) > 2 else "",
                "raw_response": response,
                "response_time": elapsed
            }
        
        results.append(result)
    
    # Show results
    print(f"\n📊 TEST RESULTS")
    print("=" * 30)
    
    successful_results = [r for r in results if r["predicted_label"] != "ERROR"]
    
    if successful_results:
        true_labels = [r["true_label"] for r in successful_results]
        pred_labels = [r["predicted_label"] for r in successful_results]
        
        accuracy = accuracy_score(true_labels, pred_labels)
        avg_time = total_time / len(results)
        
        print(f"Accuracy: {accuracy:.1%} ({len(successful_results)}/{len(results)} successful)")
        print(f"Average response time: {avg_time:.1f}s")
        print(f"Total time: {total_time:.1f}s")
        
        # Individual results
        print(f"\nIndividual Results:")
        for i, result in enumerate(results, 1):
            if result["predicted_label"] == "ERROR":
                print(f"  {i}: ERROR")
            else:
                status = "✅" if result["true_label"] == result["predicted_label"] else "❌"
                print(f"  {i}: {result['true_label']} → {result['predicted_label']} {status}")
                
        # Confusion analysis
        if len(set(pred_labels)) > 1:
            print(f"\nLabel Distribution:")
            print(f"  True: {dict(pd.Series(true_labels).value_counts())}")
            print(f"  Predicted: {dict(pd.Series(pred_labels).value_counts())}")
    
    else:
        print("❌ No successful predictions!")
    
    print(f"\n✅ Direct model test completed!")
    print(f"Prompts and responses shown above for debugging.")
    
    return len(successful_results) > 0

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("🧪 DIRECT MODEL TESTER FOR COGNITIVE DISSONANCE")
    print("=" * 60)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"🚀 CUDA available: {torch.cuda.get_device_name()}")
        print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        print("⚠️  CUDA not available, using CPU (will be slower)")
    
    success = test_direct_model()
    
    if success:
        print(f"\n🎉 Direct model test completed successfully!")
        print(f"✅ Prompt formatting works")
        print(f"✅ Model generation works") 
        print(f"✅ Response parsing works")
        print(f"\nYou can now adapt this for your server environment.")
    else:
        print(f"\n⚠️  Test had issues. Check the results and model configuration.")
        print(f"\n💡 Tips:")
        print(f"  1. Make sure MODEL_PATH points to a valid model")
        print(f"  2. Check if you have enough GPU/CPU memory")
        print(f"  3. Try a smaller model if running out of memory")
        print(f"  4. Some models may need different loading parameters")