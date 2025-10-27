#!/usr/bin/env python3
"""
Notion Auto Logger - Automatically logs prompts and outputs to daily Notion pages
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

class NotionAutoLogger:
    def __init__(self, config_file: str = "notion_logger_config.json"):
        """Initialize the Notion Auto Logger with configuration."""
        self.config_file = config_file
        self.config = self.load_config()
        self.base_dir = Path.home() / "notion_logs"
        self.base_dir.mkdir(exist_ok=True)

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default."""
        default_config = {
            "enabled": True,
            "log_prompts": True,
            "log_outputs": True,
            "auto_create_pages": True,
            "page_format": "daily",
            "timezone": "UTC",
            "categories": {
                "research": "AI Telemetry & Quantum Research",
                "coding": "Programming & Development",
                "analysis": "Data Analysis & Reports",
                "general": "General Conversation"
            }
        }

        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                print(f"Error loading config, using defaults: {e}")

        return default_config

    def save_config(self):
        """Save current configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")

    def get_daily_filename(self) -> str:
        """Generate daily log filename."""
        today = datetime.now().strftime("%Y-%m-%d")
        return f"prompts_outputs_{today}.json"

    def detect_category(self, prompt: str) -> str:
        """Detect category based on prompt content."""
        prompt_lower = prompt.lower()

        if any(word in prompt_lower for word in ['research', 'quantum', 'thesis', 'analysis', 'study']):
            return "research"
        elif any(word in prompt_lower for word in ['code', 'python', 'script', 'function', 'develop']):
            return "coding"
        elif any(word in prompt_lower for word in ['analyze', 'report', 'data', 'statistics']):
            return "analysis"
        else:
            return "general"

    def log_interaction(self, prompt: str, output: str, model: str = "claude", session_id: str = None):
        """Log a prompt-output interaction to daily file."""
        if not self.config.get("enabled", True):
            return

        timestamp = datetime.now(timezone.utc).isoformat()
        category = self.detect_category(prompt)

        log_entry = {
            "timestamp": timestamp,
            "session_id": session_id,
            "model": model,
            "category": category,
            "prompt": prompt,
            "output": output,
            "prompt_length": len(prompt),
            "output_length": len(output)
        }

        # Append to daily file
        daily_file = self.base_dir / self.get_daily_filename()

        try:
            # Load existing data or create new list
            if daily_file.exists():
                with open(daily_file, 'r') as f:
                    daily_data = json.load(f)
            else:
                daily_data = {
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "created_at": timestamp,
                    "total_interactions": 0,
                    "interactions": []
                }

            # Add new interaction
            daily_data["interactions"].append(log_entry)
            daily_data["total_interactions"] += 1
            daily_data["last_updated"] = timestamp

            # Save updated data
            with open(daily_file, 'w') as f:
                json.dump(daily_data, f, indent=2)

            print(f"✅ Logged interaction to {daily_file.name}")

        except Exception as e:
            print(f"❌ Error logging interaction: {e}")

    def get_daily_stats(self, date: str = None) -> Dict[str, Any]:
        """Get statistics for a specific date."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        filename = f"prompts_outputs_{date}.json"
        daily_file = self.base_dir / filename

        if not daily_file.exists():
            return {"error": "No data found for this date"}

        try:
            with open(daily_file, 'r') as f:
                data = json.load(f)

            # Calculate statistics
            interactions = data.get("interactions", [])
            categories = {}
            total_chars = 0

            for interaction in interactions:
                cat = interaction.get("category", "unknown")
                categories[cat] = categories.get(cat, 0) + 1
                total_chars += interaction.get("prompt_length", 0) + interaction.get("output_length", 0)

            return {
                "date": date,
                "total_interactions": len(interactions),
                "categories": categories,
                "total_characters": total_chars,
                "first_interaction": interactions[0]["timestamp"] if interactions else None,
                "last_interaction": interactions[-1]["timestamp"] if interactions else None
            }

        except Exception as e:
            return {"error": f"Error reading data: {e}"}

    def export_for_notion(self, date: str = None) -> str:
        """Export daily data in Notion-friendly format."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        stats = self.get_daily_stats(date)
        if "error" in stats:
            return f"No data found for {date}"

        notion_content = f"""# Prompt & Output Log - {date}

## Summary
- **Total Interactions:** {stats['total_interactions']}
- **Total Characters:** {stats['total_characters']:,}
- **Categories:** {', '.join([f"{k} ({v})" for k, v in stats['categories'].items()])}

## Interactions

"""

        # Load detailed interactions
        filename = f"prompts_outputs_{date}.json"
        daily_file = self.base_dir / filename

        try:
            with open(daily_file, 'r') as f:
                data = json.load(f)

            for i, interaction in enumerate(data.get("interactions", []), 1):
                timestamp = interaction["timestamp"][:19].replace("T", " ")
                category = interaction["category"]
                prompt = interaction["prompt"][:200] + "..." if len(interaction["prompt"]) > 200 else interaction["prompt"]
                output = interaction["output"][:300] + "..." if len(interaction["output"]) > 300 else interaction["output"]

                notion_content += f"""### {i}. {category.title()} - {timestamp}

**Prompt:**
{prompt}

**Output:**
{output}

---
"""

            return notion_content

        except Exception as e:
            return f"Error exporting data: {e}"

def main():
    """Command line interface for the logger."""
    if len(sys.argv) < 2:
        print("Usage: python notion_auto_logger.py <command> [args]")
        print("Commands: log, stats, export")
        return

    logger = NotionAutoLogger()
    command = sys.argv[1]

    if command == "log":
        if len(sys.argv) < 3:
            print("Usage: python notion_auto_logger.py log '<prompt>' '<output>'")
            return
        prompt = sys.argv[2]
        output = sys.argv[3] if len(sys.argv) > 3 else ""
        logger.log_interaction(prompt, output)

    elif command == "stats":
        date = sys.argv[2] if len(sys.argv) > 2 else None
        stats = logger.get_daily_stats(date)
        print(json.dumps(stats, indent=2))

    elif command == "export":
        date = sys.argv[2] if len(sys.argv) > 2 else None
        notion_content = logger.export_for_notion(date)
        print(notion_content)

    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()