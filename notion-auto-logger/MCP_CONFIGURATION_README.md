# MCP Configuration Files Documentation

This directory contains the MCP (Model Context Protocol) configuration files for Claude Code setup.

## Files Included

### 1. `claude_mcp_config.json`
**Complete MCP Server Configuration** (masked for security)

Contains the full MCP server setup including:
- **parallel-task-mcp**: HTTP-based task management with API authentication
- **parallel-search-mcp**: HTTP-based search capabilities (currently disabled)
- **perplexity**: Stdio-based deep research and search
- **filesystem**: Local file system access for `/data/data/com.termux/files/home`
- **notionApi**: Notion workspace integration for page management
- **context7**: HTTP-based documentation and library lookup

### 2. `claude_settings.json`
**Basic Claude Settings**

Core Claude Code configuration:
- Plugin system enabled for example-skills
- Always thinking mode enabled
- User preferences and UI settings

### 3. `known_marketplaces.json`
**Plugin Marketplace Configuration**

Configured marketplace sources:
- **anthropic-agent-skills**: Main skills marketplace from Anthropic
- Installation location and update timestamps

## MCP Server Details

### Active Servers
1. **Parallel Task MCP** (`https://task-mcp.parallel.ai/mcp`)
   - Deep research tasks with Pro/Ultra processors
   - API key authentication required

2. **Perplexity MCP** (`perplexity-mcp`)
   - Web search and research capabilities
   - Requires Perplexity API key

3. **Filesystem MCP** (`@modelcontextprotocol/server-filesystem`)
   - Local file system access
   - Root directory: `/data/data/com.termux/files/home`

4. **Notion API MCP** (`@notionhq/notion-mcp-server`)
   - Notion workspace integration
   - Page creation, editing, and block management

5. **Context7 MCP** (`https://mcp.context7.com/mcp`)
   - Documentation lookup and library access
   - LaTeX, programming, and technical documentation

### Disabled Servers
- **parallel-search-mcp**: Currently disabled in configuration

## Security Notes

⚠️ **Important Security Information:**
- All API keys and sensitive tokens are masked in the repository version
- Original configuration files contain actual API keys and should never be committed
- Environment variables are used for sensitive authentication data

## Installation Instructions

To restore this MCP configuration:

1. **Copy the configuration files:**
   ```bash
   cp claude_mcp_config.json ~/.claude.json
   cp claude_settings.json ~/.claude/settings.json
   cp known_marketplaces.json ~/.claude/plugins/known_marketplaces.json
   ```

2. **Update API keys** in `~/.claude.json`:
   - Replace masked `PERPLEXITY_API_KEY` with actual key
   - Replace masked `NOTION_TOKEN` with actual token
   - Replace masked `CONTEXT7_API_KEY` with actual key
   - Parallel AI API key is already included (non-sensitive)

3. **Install required npm packages:**
   ```bash
   npm install -g perplexity-mcp
   npm install -g @notionhq/notion-mcp-server
   npm install -g @modelcontextprotocol/server-filesystem
   ```

4. **Verify MCP servers:**
   ```bash
   claude mcp list
   ```

## Configuration Structure

### MCP Server Types
- **HTTP**: Remote servers via HTTP/HTTPS with API authentication
- **Stdio**: Local servers running as subprocesses
- **Environment**: Configuration through environment variables

### Server Parameters
- `type`: Connection type (http/stdio)
- `url`: HTTP endpoint URL
- `command`: Executable command for stdio servers
- `args`: Command arguments
- `env`: Environment variables
- `headers`: HTTP headers for authentication

## Troubleshooting

### Common Issues
1. **MCP servers not starting**: Check npm package installation
2. **Authentication failures**: Verify API keys are correctly set
3. **Filesystem access**: Ensure correct root directory permissions
4. **Notion integration**: Verify token has proper workspace permissions

### Debug Commands
```bash
# List all MCP servers
claude mcp list

# Check MCP server status
claude mcp status

# Test specific server
claude mcp test <server-name>
```

## Integration with Notion Auto-Logger

The MCP configuration supports the notion-auto-logger skill by providing:
- **Notion API access** for page creation and content management
- **Filesystem access** for reading/writing log files
- **Context7 access** for LaTeX documentation and formatting guidance
- **Parallel AI access** for comprehensive research capabilities

This configuration enables the complete automated logging system to function seamlessly with Claude Code's MCP architecture.

## Version History

- **v1.0** - Initial configuration setup (October 27, 2025)
- Includes all 5 active MCP servers
- Masked sensitive information for repository safety
- Complete documentation for restoration and troubleshooting