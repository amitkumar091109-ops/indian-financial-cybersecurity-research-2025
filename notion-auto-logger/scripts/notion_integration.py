#!/usr/bin/env python3
"""
Notion Integration - Connects daily logs to Notion database/pages
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

class NotionIntegration:
    def __init__(self, database_id: str = None, parent_page_id: str = None):
        """Initialize Notion integration."""
        self.database_id = database_id
        self.parent_page_id = parent_page_id
        self.base_dir = Path.home() / "notion_logs"

    def create_daily_page_content(self, date: str = None) -> str:
        """Generate Notion page content for daily logs."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        # Import the auto logger to get data
        sys.path.append(str(Path(__file__).parent))
        from notion_auto_logger import NotionAutoLogger

        logger = NotionAutoLogger()
        content = logger.export_for_notion(date)

        return content

    def format_for_notion_blocks(self, content: str) -> List[Dict]:
        """Convert markdown content to Notion block format."""
        blocks = []
        lines = content.split('\n')

        for line in lines:
            if line.startswith('# '):
                blocks.append({
                    "type": "heading_1",
                    "heading_1": {
                        "rich_text": [{"type": "text", "text": {"content": line[2:]}}]
                    }
                })
            elif line.startswith('## '):
                blocks.append({
                    "type": "heading_2",
                    "heading_2": {
                        "rich_text": [{"type": "text", "text": {"content": line[3:]}}]
                    }
                })
            elif line.startswith('### '):
                blocks.append({
                    "type": "heading_3",
                    "heading_3": {
                        "rich_text": [{"type": "text", "text": {"content": line[4:]}}]
                    }
                })
            elif line.startswith('- '):
                blocks.append({
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [{"type": "text", "text": {"content": line[2:]}}]
                    }
                })
            elif line.startswith('**') and line.endswith('**'):
                blocks.append({
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{
                            "type": "text",
                            "text": {"content": line[2:-2]},
                            "annotations": {"bold": True}
                        }]
                    }
                })
            elif line.strip() == '':
                blocks.append({
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": ""}}]
                    }
                })
            elif line.strip() == '---':
                blocks.append({
                    "type": "divider",
                    "divider": {}
                })
            else:
                blocks.append({
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": line}}]
                    }
                })

        return blocks

def main():
    """Command line interface for Notion integration."""
    integration = NotionIntegration()

    # Example: Create content for today
    today = datetime.now().strftime("%Y-%m-%d")
    content = integration.create_daily_page_content(today)

    print("Generated Notion content:")
    print("=" * 50)
    print(content)
    print("=" * 50)

    # Convert to blocks
    blocks = integration.format_for_notion_blocks(content)
    print(f"\nGenerated {len(blocks)} Notion blocks")

    # Save blocks to file for manual Notion import
    with open(integration.base_dir / f"notion_blocks_{today}.json", 'w') as f:
        json.dump(blocks, f, indent=2)

    print(f"Blocks saved to: {integration.base_dir}/notion_blocks_{today}.json")

if __name__ == "__main__":
    import sys
    main()