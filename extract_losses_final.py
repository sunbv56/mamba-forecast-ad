import os
import json
import re
import argparse

def parse_notebook_losses(notebook_path, save_markdown_path=None):
    """
    Parses a Jupyter Notebook (.ipynb) to extract training and validation losses
    for each epoch, filtering out progress bar updates (tqdm).
    """
    if not os.path.exists(notebook_path):
        print(f"Error: File not found at '{notebook_path}'")
        return None

    print(f"Reading notebook from: {notebook_path}...")
    with open(notebook_path, "r", encoding="utf-8") as f:
        try:
            nb = json.load(f)
        except Exception as e:
            print(f"Error parsing JSON: {e}")
            return None

    epoch_logs = []
    print("Parsing cell outputs...")
    current_model = "Unknown Model"

    for cell_idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        
        outputs = cell.get("outputs", [])
        for output in outputs:
            lines = []
            if output.get("output_type") == "stream":
                text_data = output.get("text", [])
                if isinstance(text_data, list):
                    lines.extend(text_data)
                elif isinstance(text_data, str):
                    lines.extend(text_data.splitlines(keepends=True))
            elif "data" in output and "text/plain" in output["data"]:
                text_data = output["data"]["text/plain"]
                if isinstance(text_data, list):
                    lines.extend(text_data)
                elif isinstance(text_data, str):
                    lines.extend(text_data.splitlines(keepends=True))
            
            for line in lines:
                line_str = line.strip()
                if not line_str:
                    continue
                
                # Check for model name updates from any line (including tqdm lines)
                tqdm_model_match = re.search(r"^([A-Za-z0-9_-]+)\s*-\s*Epoch\s+\d+", line_str, re.IGNORECASE)
                if tqdm_model_match:
                    current_model = tqdm_model_match.group(1)
                else:
                    model_name_match = re.search(r"(?:Huấn luyện|DataLoader cho)\s+([A-Za-z0-9_-]+)", line_str, re.IGNORECASE)
                    if model_name_match:
                        current_model = model_name_match.group(1)
                
                # Skip progress bar lines from tqdm for loss logging
                if "|" in line_str and ("%" in line_str or "it/s" in line_str or "s/it" in line_str):
                    continue
                
                # Detect train or validation loss logs
                if "epoch" in line_str.lower() and "loss" in line_str.lower():
                    epoch_logs.append((current_model, cell_idx, line_str))

    if not epoch_logs:
        print("No matching training/validation loss logs found.")
        return None

    print(f"Successfully extracted {len(epoch_logs)} logs.")
    
    # Process logs into a structured table grouped by (model, epoch)
    records = {}
    for model, cell_idx, log in epoch_logs:
        # Extract epoch number
        epoch_match = re.search(r"epoch\s*\[?(\d+)/(\d+)\]?", log, re.IGNORECASE)
        if not epoch_match:
            epoch_match = re.search(r"epoch\s*(\d+)", log, re.IGNORECASE)
            
        if epoch_match:
            epoch_num = int(epoch_match.group(1))
            key = (model, epoch_num)
            if key not in records:
                records[key] = {"train": "N/A", "val": "N/A"}
                
            loss_match = re.search(r"(train|val|validation)\s*loss:\s*([0-9.]+)", log, re.IGNORECASE)
            if loss_match:
                loss_type = loss_match.group(1).lower()
                loss_val = float(loss_match.group(2))
                
                if "train" in loss_type:
                    records[key]["train"] = f"{loss_val:.6f}"
                elif "val" in loss_type or "validation" in loss_type:
                    records[key]["val"] = f"{loss_val:.6f}"

    # Get ordered model list to maintain sequence
    seen_models = []
    for model, _, _ in epoch_logs:
        if model not in seen_models:
            seen_models.append(model)
            
    # Sort keys by model sequence index, then epoch number
    sorted_keys = sorted(records.keys(), key=lambda k: (seen_models.index(k[0]), k[1]))
    
    table_data = []
    for model, epoch in sorted_keys:
        table_data.append((model, epoch, records[(model, epoch)]["train"], records[(model, epoch)]["val"]))

    # Print results formatted
    print("\n" + "="*75)
    print(f"      EXTRACTED LOSS HISTORY")
    print("="*75)
    print(f"{'Model':<20} | {'Epoch':<10} | {'Train Loss':<15} | {'Val Loss':<15}")
    print("-" * 75)
    for model, epoch, t_loss, v_loss in table_data:
        print(f"{model:<20} | {epoch:<10} | {t_loss:<15} | {v_loss:<15}")
    print("="*75)

    # Save to file if path provided
    if save_markdown_path:
        os.makedirs(os.path.dirname(save_markdown_path), exist_ok=True)
        with open(save_markdown_path, "w", encoding="utf-8") as out_f:
            out_f.write("# Extracted Loss History\n\n")
            out_f.write(f"Parsed from: `{notebook_path}`\n\n")
            out_f.write("| Model | Epoch | Train Loss | Val Loss |\n")
            out_f.write("| :--- | :---: | :---: | :---: |\n")
            for model, epoch, t_loss, v_loss in table_data:
                out_f.write(f"| **{model}** | {epoch} | {t_loss} | {v_loss} |\n")
            
            out_f.write("\n## Raw Log Lines\n\n")
            out_f.write("```text\n")
            for model, cell_idx, log in epoch_logs:
                out_f.write(f"[{model}][Cell {cell_idx}] {log}\n")
            out_f.write("```\n")
        print(f"\nSaved structured markdown report to: {save_markdown_path}")

    return table_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Epoch losses from large Jupyter Notebook.")
    parser.add_argument("--notebook", default=r"c:\Users\Acer\Downloads\train-test-final-mamba-forecast-ad.ipynb", help="Path to .ipynb file")
    parser.add_argument("--output", default=r"f:\APPS_PJ\mamba-forecast-ad\results\extracted_loss_history.md", help="Path to save markdown report")
    args = parser.parse_args()

    parse_notebook_losses(args.notebook, args.output)
