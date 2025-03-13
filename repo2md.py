
import os
import re
import argparse
import requests
from git import Repo, GitCommandError
from pathlib import Path
from typing import List, Optional
from slugify import slugify
import mimetypes

class RepoToMDConverter:
    def __init__(self, 
                 repo_url: str, 
                 output_file: str = "output.md",
                 exclude_dirs: Optional[List[str]] = None,
                 max_file_size: int = 1024*1024,
                 fetch_metadata: bool = True,
                 cleanup: bool = True):
        
        self.repo_url = repo_url
        self.output_file = output_file
        self.exclude_dirs = exclude_dirs or ['.git', 'node_modules', '__pycache__', 'venv']
        self.max_file_size = max_file_size
        self.fetch_metadata = fetch_metadata
        self.cleanup_enabled = cleanup
        self.repo_path = None
        self.repo_name = self.extract_repo_name()
        self.toc = []
        self.content = []
        self.metadata = {}
        self.tree_content = ""

    def extract_repo_name(self) -> str:
        """Extract repository name from URL"""
        match = re.search(r"github.com[/:](.+?)/(.+?)(?:\.git|$)", self.repo_url)
        return f"{match.group(1)}/{match.group(2)}" if match else "repository"

    def fetch_github_metadata(self):
        """Fetch repository metadata from GitHub API"""
        api_url = f"https://api.github.com/repos/{self.repo_name}"
        try:
            response = requests.get(api_url, headers={'Accept': 'application/vnd.github.v3+json'})
            if response.ok:
                data = response.json()
                self.metadata = {
                    'description': data.get('description', ''),
                    'owner': data.get('owner', {}).get('login', ''),
                    'stars': data.get('stargazers_count', 0),
                    'forks': data.get('forks_count', 0),
                    'created_at': data.get('created_at', ''),
                    'updated_at': data.get('updated_at', ''),
                    'license': data.get('license', {}).get('name', '')
                }
        except requests.exceptions.RequestException:
            pass

    def clone_repository(self):
        """Clone the repository to a temporary directory"""
        self.repo_path = os.path.join(os.getcwd(), "temp_repo")
        Repo.clone_from(self.repo_url, self.repo_path)

    def is_binary_file(self, file_path: str) -> bool:
        """Check if a file is binary"""
        mime = mimetypes.guess_type(file_path)[0]
        if mime and not mime.startswith('text'):
            return True
        
        try:
            with open(file_path, 'rb') as f:
                return b'\x00' in f.read(1024)
        except IOError:
            return True

    def get_code_language(self, filename: str) -> str:
        """Map file extension to Markdown language"""
        extension_map = {
            'py': 'python', 'js': 'javascript', 'ts': 'typescript',
            'java': 'java', 'c': 'c', 'cpp': 'cpp', 'h': 'c',
            'html': 'html', 'css': 'css', 'md': 'markdown',
            'json': 'json', 'yml': 'yaml', 'yaml': 'yaml',
            'sh': 'bash', 'go': 'go', 'rs': 'rust', 'php': 'php',
            'rb': 'ruby', 'kt': 'kotlin', 'swift': 'swift'
        }
        return extension_map.get(filename.split('.')[-1].lower(), '')

    def generate_directory_tree(self):
        """Generate ASCII directory tree structure"""
        tree_lines = ["```"]
        
        def walk_directory(current_path, prefix="", is_last=False):
            try:
                entries = sorted(os.listdir(current_path))
            except PermissionError:
                return

            filtered = []
            for entry in entries:
                entry_path = os.path.join(current_path, entry)
                if os.path.isdir(entry_path):
                    if entry in self.exclude_dirs:
                        continue
                    filtered.append( (entry_path, True) )
                else:
                    if any(x in entry_path for x in self.exclude_dirs):
                        continue
                    if os.path.getsize(entry_path) > self.max_file_size:
                        continue
                    if self.is_binary_file(entry_path):
                        continue
                    filtered.append( (entry_path, False) )

            for i, (path, is_dir) in enumerate(filtered):
                is_last_entry = i == len(filtered) - 1
                name = os.path.basename(path) + ("/" if is_dir else "")
                
                if is_last:
                    connector = "└── " if is_last_entry else "├── "
                    new_prefix = "    " if is_last_entry else "│   "
                else:
                    connector = "└── " if is_last_entry else "├── "
                    new_prefix = "│   " if not is_last_entry else "    "

                tree_lines.append(f"{prefix}{connector}{name}")
                
                if is_dir:
                    walk_directory(
                        path,
                        prefix + new_prefix,
                        is_last_entry
                    )

        walk_directory(self.repo_path)
        tree_lines.append("```")
        self.tree_content = "\n".join(tree_lines)

    def generate_header(self):
        """Generate Markdown header section"""
        header = [f"# {self.repo_name}\n"]
        
        if self.metadata:
            header.extend([
                f"**Description**: {self.metadata.get('description', '')}  ",
                f"**Owner**: [{self.metadata.get('owner', '')}](https://github.com/{self.metadata.get('owner', '')})  ",
                f"⭐ Stars: {self.metadata.get('stars', 0)} | 🍴 Forks: {self.metadata.get('forks', 0)}  ",
                f"**Created**: {self.metadata.get('created_at', '')}  ",
                f"**Last Updated**: {self.metadata.get('updated_at', '')}  ",
                f"**License**: {self.metadata.get('license', '')}  \n"
            ])
        
        header.append(f"GitHub URL: [{self.repo_url}]({self.repo_url})  \n")
        return "\n".join(header)

    def process_directory(self):
        """Process repository files and build content"""
        for root, dirs, files in os.walk(self.repo_path):
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]
            rel_path = os.path.relpath(root, self.repo_path)
            section = "Root Directory" if rel_path == "." else rel_path
            
            self.content.append(f"\n## `{section}`\n")
            
            for file in files:
                file_path = os.path.join(root, file)
                if any(x in file_path for x in self.exclude_dirs):
                    continue
                
                if os.path.getsize(file_path) > self.max_file_size:
                    continue
                
                if self.is_binary_file(file_path):
                    continue

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except UnicodeDecodeError:
                    continue

                anchor = slugify(f"{rel_path}-{file}")
                self.toc.append(f"- [`{file}`](#{anchor})")
                
                language = self.get_code_language(file)
                self.content.append(
                    f"\n### {file} {{#{anchor}}}\n"
                    f"```{language}\n{content}\n```\n"
                )

    def generate_output(self):
        """Generate final Markdown output"""
        header = self.generate_header()
        toc_section = "## Table of Contents\n" + "\n".join(self.toc)
        
        final_content = (
            f"{header}\n"
            f"## Directory Structure\n{self.tree_content}\n\n"
            f"{toc_section}\n"
            f"{''.join(self.content)}"
        )
        
        Path(self.output_file).write_text(final_content, encoding='utf-8')

    def cleanup(self):
        """Clean up temporary files"""
        if self.repo_path and os.path.exists(self.repo_path):
            import shutil
            shutil.rmtree(self.repo_path)

    def convert(self):
        """Main conversion workflow"""
        try:
            self.clone_repository()
            if self.fetch_metadata:
                self.fetch_github_metadata()
            self.generate_directory_tree()
            self.process_directory()
            self.generate_output()
        finally:
            if self.cleanup_enabled:
                self.cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a GitHub repository to structured Markdown documentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--repo-url", required=True, help="GitHub repository URL")
    parser.add_argument("--output", default="output.md", help="Output markdown file path")
    parser.add_argument("--exclude-dirs", nargs="+", default=['.git', 'node_modules', '__pycache__'],
                      help="Directories to exclude")
    parser.add_argument("--max-file-size", type=int, default=1024*1024,
                      help="Maximum file size in bytes to include")
    parser.add_argument("--no-metadata", action="store_false", dest="fetch_metadata",
                      help="Disable fetching GitHub metadata")
    parser.add_argument("--no-cleanup", action="store_false", dest="cleanup",
                      help="Keep temporary repository clone")
    args = parser.parse_args()

    converter = RepoToMDConverter(
        repo_url=args.repo_url,
        output_file=args.output,
        exclude_dirs=args.exclude_dirs,
        max_file_size=args.max_file_size,
        fetch_metadata=args.fetch_metadata,
        cleanup=args.cleanup
    )
    converter.convert()
    print(f"Successfully generated documentation: {args.output}")