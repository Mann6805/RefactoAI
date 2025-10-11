# src/inference.py

import sys
import torch
import os
import ast
import javalang
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn import Sigmoid

# ----------------------------
# Configuration
# ----------------------------
MODEL_PATH = "models/graphcodebert_finetuned/"
MAX_LEN = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFECT_TYPES = ["resource_leak", "null_pointer_dereference", "concurrency_issue", 
                "security_vulnerability", "code_complexity"]

# ----------------------------
# Load model & tokenizer
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()
sigmoid = Sigmoid()

# ----------------------------
# Python function extraction using AST
# ----------------------------
def extract_python_functions(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        code_lines = f.readlines()
        tree = ast.parse("".join(code_lines))

    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            start = node.lineno - 1
            end = node.body[-1].lineno  # last line of function
            func_code = "".join(code_lines[start:end])
            functions.append(func_code)
    if not functions:
        # If no functions, treat whole file as one
        with open(file_path, "r", encoding="utf-8") as f:
            functions = [f.read()]
    return functions

# ----------------------------
# Java method extraction using javalang
# ----------------------------
def extract_java_methods(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        code = f.read()
    methods = []

    tree = javalang.parse.parse(code)
    lines = code.splitlines()
    for path, node in tree:
        if isinstance(node, javalang.tree.MethodDeclaration):
            if node.position:
                start = node.position.line - 1
                # Approximate end by counting braces (naive but works for simple methods)
                brace_count = 0
                end = start
                for i in range(start, len(lines)):
                    brace_count += lines[i].count("{")
                    brace_count -= lines[i].count("}")
                    end = i
                    if brace_count == 0:
                        break
                method_code = "\n".join(lines[start:end+1])
                methods.append(method_code)
    if not methods:
        methods = [code]
    return methods

# ----------------------------
# Predict defects
# ----------------------------
def predict_defects(functions):
    results = []
    for idx, func in enumerate(functions):
        inputs = tokenizer(func, padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k,v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = sigmoid(logits).cpu().numpy()[0]
        defects = [DEFECT_TYPES[i] for i, p in enumerate(probs) if p >= 0.5]
        results.append({
            "function_index": idx+1,
            "code": func,
            "defects": defects,
            "defect_probs": probs.tolist()
        })
    return results

# ----------------------------
# Display results
# ----------------------------
def display_results(results):
    for res in results:
        print("="*60)
        print(f"Function #{res['function_index']}:")
        print(res['code'])
        if res["defects"]:
            print(f"\n🚨 Defects detected: {', '.join(res['defects'])}")
            for d, prob in zip(DEFECT_TYPES, res["defect_probs"]):
                if prob >= 0.5:
                    print(f"  - {d}: probability {prob:.3f}")
        else:
            print("\n✅ No defects detected")
        print("="*60 + "\n")

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inference.py <path_to_code_file.py_or.java>")
        sys.exit(1)

    file_path = sys.argv[1]
    if not os.path.isfile(file_path):
        print("File not found:", file_path)
        sys.exit(1)

    # Detect language
    ext = os.path.splitext(file_path)[1].lower()
    language = "python" if ext == ".py" else "java" if ext == ".java" else None
    if not language:
        print("Unsupported file extension. Only .py and .java allowed.")
        sys.exit(1)

    # Extract functions/methods
    if language == "python":
        functions = extract_python_functions(file_path)
    else:
        functions = extract_java_methods(file_path)

    print(f"Found {len(functions)} function(s)/method(s) in the file.")

    # Predict defects
    results = predict_defects(functions)

    # Display
    display_results(results)