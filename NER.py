import os
import re
import json
import pandas as pd
from collections import defaultdict
import unicodedata


# Load data from external files
try:
    drug_data = pd.read_csv("corpus/Drug_list_04_17_31_2024.csv")
    cell_lines = pd.read_csv("corpus/cell-line-selector.csv")["displayName"].dropna().tolist()
    assay_data_path = "corpus/assay_data.json"

    # Load assay data
    with open(assay_data_path, 'r') as json_file:
        assay_data = json.load(json_file)

    # Synonym to drug name mapping
    synonym_to_name = {
        syn.strip().lower(): row["Name"]
        for _, row in drug_data.iterrows()
        for syn in str(row["Synonyms"]).split(";")
    }

    # Preprocess assay_data to compile regex patterns
    # Safely preprocess regex patterns in assay_data
    for assay, assay_info in assay_data.items():
        if "parameters" in assay_info:
            for key, patterns in assay_info["parameters"].items():
                # Convert stored patterns into precompiled regex
                compiled_patterns = [
                    re.compile(pattern.replace("\\\\", "\\")) for pattern in patterns
                ]
                # Store precompiled regex in a new field
                assay_info["parameters"][key] = compiled_patterns



except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Preprocess content to limit between "abstract" and "references"
def preprocess_content(content):
    # Step 1: Normalize Unicode to clean up garbled characters
    content = unicodedata.normalize('NFKC', content)

    # Step 2: Systematic fix for common garbled encodings using a mapping
    garbled_mappings = {
        "‚Äâ": "",      # Thin space
        "√ó": "×",       # Multiplication sign
        "‚Äî": "-",      # Em dash
        "‚Äô": "'",      # Apostrophe
        "‚Äì": "-",      # En dash
        "‚Ä¶": "...",    # Ellipsis
    }

    # Use regex to replace all garbled mappings
    for garbled, replacement in garbled_mappings.items():
        content = content.replace(garbled, replacement)

    content = re.sub(
        r'(\b(?:\d+(?:\.\d+)?|e))\s+(\d+)\b',  # Regex for numbers or "e" followed by spaces and digits
        lambda m: f"{m.group(1)}{''.join(chr(8304 + int(d)) for d in m.group(2))}",
        content
    )

    # Step 2: Convert content to lowercase for easier searching
    content_lower = content.lower()

    # Calculate the indexes for one-fourth and two-thirds of the content length
    one_fourth_length = len(content_lower) * 1 // 4
    two_thirds_length = len(content_lower) * 2 // 3

    # Find the start index of "abstract"
    start_index = content_lower[:one_fourth_length].rfind("abstract")

    # Find the end index of "references"
    match = re.search(r'\breferences\b', content_lower[two_thirds_length:], re.IGNORECASE)
    end_index = two_thirds_length + match.start() if match else -1

    # Trim content based on "abstract" and "references"
    if start_index != -1:
        content = content[start_index + len("abstract"):]
    if end_index != -1:
        content = content[:end_index]

    lines = content.splitlines()
    cleaned_lines = []
    seen_lines = set()

    for i, line in enumerate(lines):
        if line == "" or line in seen_lines:
            continue  # Skip blank lines
        if i > 0 and not re.search(r'[a-zA-Z][.!?"]$', lines[i-1].strip()) and not re.search(r'[^\w\s][.!?"]$', lines[i-1].strip()):
        # Remove newlines if the previous line doesn't end with a proper sentence-ending period
            #print("condition 1 met: ", lines[i-1].strip())
            #print("condition 2 met: ", lines[i-1].strip().istitle() )
            #print("condition 2 met: ", lines[i].strip()[:1].isupper() )
            seen_lines.add(line)
            if i < len(lines) - 1 and not (lines[i-1].strip().istitle() and lines[i].strip()[:1].isupper()):
            # Remove newlines if the current line isn't a title and the next line isn't a capital letter
                if cleaned_lines:
                    previous = cleaned_lines.pop()
                    line = previous + " " + line

        if line not in seen_lines:
            cleaned_lines.append(line)

    # Join the cleaned lines
    content = "\n".join(cleaned_lines)

    # Add newline to all sentence-end positions
    content = re.sub(r'(?<=[a-zA-Z.!?\]”\)])[.!?\]”\)]\s(?=[A-Z0-9]|[.!?\]”\)])', '\g<0>\n', content)

    # Return the cleaned and modified content
    return content.strip()

def extract_entities(content, assay_data, synonym_to_name, cell_lines, pmid, results):
    lines = content.splitlines()
    drug_mentions = defaultdict(list)
    cell_line_mentions = defaultdict(list)
    assays_found = defaultdict(lambda: defaultdict(list))

    for line_number, line in enumerate(lines):
        line_entities = []
        reference_pattern = r'\d{4};\d+\(\d+\):|https?://\S+|([A-Z][a-z]+,? ?)+([A-Z]\.)+'
        if re.search(reference_pattern, line):
            continue  # Skip processing this line

        # Drugs
        drug_pattern = r'\b(?:' + '|'.join(map(re.escape, synonym_to_name.keys())) + r')\b'
        for match in re.finditer(drug_pattern, line):
            synonym = match.group()
            drug_name = synonym_to_name[synonym.lower()]
            drug_mentions[drug_name].append(match.start())
            line_entities.append({
                "entity": drug_name,
                "type": "Drug",
                "start": match.start(),
                "end": match.end()
            })

        # Cell lines
        cell_line_pattern = r'\b(?:' + '|'.join(map(re.escape, cell_lines)) + r')\b'
        for match in re.finditer(cell_line_pattern, line):
            cell_line = match.group()
            cell_line_mentions[cell_line].append(match.start())
            line_entities.append({
                "entity": cell_line,
                "type": "CellLine",
                "start": match.start(),
                "end": match.end()
            })

        # Assays
        for assay, assay_info in assay_data.items():
            if any(re.search(keyword, line, re.IGNORECASE) for keyword in assay_info["keywords"]):
                for parameter, patterns in assay_info.get("parameters", {}).items():
                    for pattern in patterns:
                        for match in pattern.finditer(line):
                            value = match.group()
                            start_pos = match.start(0)
                            end_pos = match.end(0)
                            print(f"Value: {value}, Start: {start_pos}, End: {end_pos}")
                            # text = line[start_pos:end_pos]
                            # Add found assay information
                            assays_found[assay][parameter].append(value)
                            #print(text)
                            line_entities.append({
                                "entity": value,
                                "type": f"{parameter}",
                                "start": start_pos,
                                "end": end_pos
                            })
                            print(line)


        # Append results for the current line
        if line_entities:  # Only include lines with found entities
            new_entity = {
                "pmid": pmid,
                "line_number": line_number + 1,
                "line": line,
                "entities": line_entities
            }
            results.append(new_entity)

    focused_drugs = sorted(drug_mentions.items(), key=lambda x: len(x[1]), reverse=True)[:3]
    focused_cell_lines = sorted(cell_line_mentions.items(), key=lambda x: len(x[1]), reverse=True)[:3]
    summary = {
        "pmid": pmid,
        "Drug Name": list(dict(focused_drugs).keys()),
        "Cell Line": list(dict(focused_cell_lines).keys()),
        "Assay": [{"assay": assay, "params": {key: list(set(values))[0] for key, values in params.items()}} for assay, params in assays_found.items()]
    }

    return results, summary


# Process files
folder_path = "pubmed_queries/full_outputs"
results = []
general_results = []

try:
    files = sorted(f for f in os.listdir(folder_path) if f.endswith("_content.txt"))
    for file_name in files:
        pmid = file_name.split("_")[0]
        print(pmid)
        with open(os.path.join(folder_path, file_name), 'r') as f:
            content = preprocess_content(f.read())

        results, summary = extract_entities(content, assay_data, synonym_to_name, cell_lines, pmid, results)
        general_results.append(summary)
    df = pd.DataFrame(results)
    df_sum = pd.DataFrame(general_results)
    df.to_csv("results/processed_data.csv", encoding='utf-8', index=False)
    df_sum.to_csv("results/structure.csv", encoding='utf-8', index=False)
    print("Processed data saved to 'results/processed_data.csv'")
except Exception as e:
    print(f"Error processing files: {e}")
