import os

root_dir = r"d:\CAPSTONE\capstone-2\Code"

for root, _, files in os.walk(root_dir):
    for file in files:
        if file.endswith(".py"):
            path = os.path.join(root, file)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            
            if "Code.models" in content or "Code.config" in content or "src." in content:
                # carefully replace Code.models and Code.config, and Code.interface
                new_content = content.replace("Code.models", "Code.models")
                new_content = new_content.replace("Code.config", "Code.config")
                new_content = new_content.replace("Code.interface", "Code.interface")
                new_content = new_content.replace("Code.data", "Code.data")
                new_content = new_content.replace("Code.utils", "Code.utils")
                
                if new_content != content:
                    print(f"Fixed {path}")
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(new_content)
