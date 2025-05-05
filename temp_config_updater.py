import re

def add_new_role(input_file, output_file, new_role_name):
    """
    Adds a new role configuration block after the hm-wbmgmt role
    and preserves the exact spacing and formatting of the entire file.
    
    Args:
        input_file (str): Path to the input configuration file
        output_file (str): Path to the output configuration file
        new_role_name (str): Name of the new role to add
    """
    # Read the content of the file as binary to preserve exact format
    with open(input_file, 'rb') as f:
        content_bytes = f.read()
    
    # Convert to string for processing
    content = content_bytes.decode('utf-8')
    
    # Define a pattern to match the entire hm-wbmgmt role block
    pattern = r'(aaa authorization user-role name "hm-wbmgmt".*?exit\s+exit\s+)'
    
    # Search for the pattern
    match = re.search(pattern, content, re.DOTALL)
    
    if not match:
        print("Could not find the hm-wbmgmt role section!")
        return False
    
    # Extract the original block to analyze its exact format
    wbmgmt_role_text = match.group(0)
    
    # Create the new role by copying the format of the original
    # but replacing the name
    new_role_config = wbmgmt_role_text.replace('"hm-wbmgmt"', f'"{new_role_name}"')
    new_role_config = new_role_config.replace('secondary-role "hm-wbmgmt"', f'secondary-role "{new_role_name}"')
    
    # Get the match position and insert the new role after it
    end_pos = match.end()
    updated_content = content[:end_pos] + new_role_config + content[end_pos:]
    
    # Write to output file as binary to preserve format
    with open(output_file, 'wb') as f:
        f.write(updated_content.encode('utf-8'))
    
    print(f"Successfully added new role '{new_role_name}' after hm-wbmgmt")
    return True

# Example usage
if __name__ == "__main__":
    input_file = "template_config.txt"
    output_file = "updated_template_config.txt"
    new_role_name = input("Enter the new role to be added: ")  # Change this to your desired role name
    
    add_new_role(input_file, output_file, new_role_name)