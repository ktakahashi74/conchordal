set shell := ["bash", "-euo", "pipefail", "-c"]
set dotenv-load := true

SESSION_FILE := ".chat_url"
PROMPT_FILE := ".driver_prompt.md"
SYNC_FILE := "sync_report.md"
CONTEXT_FILE := "context.xml"
DIFF_FILE := "diff.patch"
SYNC_MAX_BYTES := `bash -c 'echo ${SYNC_MAX_BYTES:-90000}'`
DIFF_BASE_MODE := `bash -c 'echo ${DIFF_BASE_MODE:-head}'`
TS := `date +"%Y%m%d-%H%M%S"`

OS := `uname -s`
IS_WSL := `bash -c 'grep -qi microsoft /proc/version /proc/sys/kernel/osrelease 2>/dev/null && echo 1 || echo 0'`
HAS_CMD := `bash -c 'command -v cmd.exe >/dev/null 2>&1 && echo 1 || echo 0'`

CLIP := `bash -c 'if [ "{{OS}}" = "Darwin" ]; then echo "pbcopy"; elif [ "{{OS}}" = "Linux" ] && [ "{{IS_WSL}}" = "1" ] && [ "{{HAS_CMD}}" = "1" ]; then echo "clip.exe"; elif [ "{{OS}}" = "Linux" ] && command -v wl-copy >/dev/null 2>&1; then echo "wl-copy"; elif [ "{{OS}}" = "Linux" ] && command -v xclip >/dev/null 2>&1; then echo "xclip -selection clipboard"; else echo "cat > /dev/null"; fi'`
PASTE := `bash -c 'if [ "{{OS}}" = "Darwin" ]; then echo "pbpaste"; elif [ "{{OS}}" = "Linux" ] && [ "{{IS_WSL}}" = "1" ] && [ "{{HAS_CMD}}" = "1" ]; then echo "powershell.exe -c Get-Clipboard"; elif [ "{{OS}}" = "Linux" ] && command -v wl-paste >/dev/null 2>&1; then echo "wl-paste"; elif [ "{{OS}}" = "Linux" ] && command -v xclip >/dev/null 2>&1; then echo "xclip -selection clipboard -o"; else echo "cat"; fi'`
OPEN := `bash -c 'if [ "{{OS}}" = "Darwin" ]; then echo "open"; elif [ "{{OS}}" = "Linux" ] && [ "{{IS_WSL}}" = "1" ] && [ "{{HAS_CMD}}" = "1" ]; then echo "cmd.exe /c start"; elif [ "{{OS}}" = "Linux" ] && command -v xdg-open >/dev/null 2>&1; then echo "xdg-open"; else echo "true"; fi'`
CHAT_URL := `bash -c "if [ -f \"{{SESSION_FILE}}\" ]; then cat \"{{SESSION_FILE}}\"; else echo \"https://chatgpt.com/\"; fi"`
REPOMIX_CMD := `bash -c 'if command -v repomix >/dev/null 2>&1; then echo "repomix"; else echo "npx repomix"; fi'`

default:
	@just --list

session url="":
	@if [ -n "{{url}}" ]; then \
		echo "{{url}}" > "{{SESSION_FILE}}"; \
	else \
		echo "https://chatgpt.com/" > "{{SESSION_FILE}}"; \
	fi
	@cat "{{SESSION_FILE}}"

pack:
	@{{REPOMIX_CMD}} --config repomix.config.json; \
	if [ -f "{{CONTEXT_FILE}}" ]; then \
		mv "{{CONTEXT_FILE}}" "context-{{TS}}.xml"; \
		ln -sf "context-{{TS}}.xml" "{{CONTEXT_FILE}}"; \
	fi; \
	printf "### git diff (unstaged)\n" > "diff-{{TS}}.patch"; \
	git diff >> "diff-{{TS}}.patch"; \
	printf "\n### git diff --cached (staged)\n" >> "diff-{{TS}}.patch"; \
	git diff --cached >> "diff-{{TS}}.patch"; \
	ln -sf "diff-{{TS}}.patch" "{{DIFF_FILE}}"; \
	{{OPEN}} .; \
	{{OPEN}} "{{CHAT_URL}}"; \
	echo "Upload context-{{TS}}.xml and diff-{{TS}}.patch (recommended)."; \
	echo "context.xml and diff.patch are stable aliases of the latest run."

diff:
	@printf "### git diff (unstaged)\n" > "diff-{{TS}}.patch"; \
	git diff >> "diff-{{TS}}.patch"; \
	printf "\n### git diff --cached (staged)\n" >> "diff-{{TS}}.patch"; \
	git diff --cached >> "diff-{{TS}}.patch"; \
	ln -sf "diff-{{TS}}.patch" "{{DIFF_FILE}}"; \
	{{OPEN}} "{{CHAT_URL}}"; \
	echo "Upload diff-{{TS}}.patch (recommended)."; \
	echo "diff.patch is a stable alias of the latest run."

sync:
	@timestamp="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"; \
	status="$(git status -sb)"; \
	diff_stat="$(git diff --stat)"; \
	diff_cached_stat="$(git diff --stat --cached)"; \
	diff_full="$(git diff)"; \
	diff_cached_full="$(git diff --cached)"; \
	risk_scan="$(rg -n 'unsafe\b|\bunwrap\(|\bexpect\(|process::exit|panic!\(|todo!\(|unimplemented!\(' -S . || true)"; \
	check_output="$(cargo check 2>&1 || true)"; \
	check_tail="$(printf "%s\n" "${check_output}" | tail -n 200)"; \
	{ \
		echo "# Sync Report"; \
		echo ""; \
		echo "Timestamp: ${timestamp}"; \
		echo ""; \
		echo "## Git Status"; \
		echo "${status}"; \
		echo ""; \
		echo "## Diff Stat (unstaged)"; \
		echo "${diff_stat}"; \
		echo ""; \
		echo "## Diff Stat (staged)"; \
		echo "${diff_cached_stat}"; \
		echo ""; \
		echo "## Diff (unstaged)"; \
		printf '%s\n' '```diff'; \
		echo "${diff_full}"; \
		printf '%s\n' '```'; \
		echo ""; \
		echo "## Diff (staged)"; \
		printf '%s\n' '```diff'; \
		echo "${diff_cached_full}"; \
		printf '%s\n' '```'; \
		echo ""; \
		echo "## Risk Scan"; \
		if [ -n "${risk_scan}" ]; then echo "${risk_scan}"; else echo "No matches."; fi; \
		echo ""; \
		if [ "{{DIFF_BASE_MODE}}" = "upstream" ] && git rev-parse --abbrev-ref --symbolic-full-name @{u} >/dev/null 2>&1; then \
			echo "## Upstream Diff Stat"; \
			git diff --stat --merge-base @{u}...HEAD; \
			echo ""; \
		fi; \
		echo "## Cargo Check (tail)"; \
		printf '%s\n' '```'; \
		echo "${check_tail}"; \
		printf '%s\n' '```'; \
	} > "{{SYNC_FILE}}"; \
	if [ "{{DIFF_BASE_MODE}}" = "upstream" ] && git rev-parse --abbrev-ref --symbolic-full-name @{u} >/dev/null 2>&1; then \
		upstream_patch="$(git diff --merge-base @{u}...HEAD)"; \
		current_size="$(wc -c < "{{SYNC_FILE}}")"; \
		patch_size="$(printf "%s" "${upstream_patch}" | wc -c)"; \
		target_size="$((current_size + patch_size))"; \
		if [ "${target_size}" -le "{{SYNC_MAX_BYTES}}" ]; then \
			{ \
				echo ""; \
				echo "## Upstream Diff (merge-base)"; \
				printf '%s\n' '```diff'; \
				echo "${upstream_patch}"; \
				printf '%s\n' '```'; \
			} >> "{{SYNC_FILE}}"; \
		fi; \
	fi; \
	file_size="$(wc -c < "{{SYNC_FILE}}")"; \
	if [ "${file_size}" -gt "{{SYNC_MAX_BYTES}}" ]; then \
		{ \
			echo "# Sync Report (Trimmed)"; \
			echo ""; \
			echo "Timestamp: ${timestamp}"; \
			echo ""; \
			echo "## Git Status"; \
			echo "${status}"; \
			echo ""; \
			echo "## Diff Stat (unstaged)"; \
			echo "${diff_stat}"; \
			echo ""; \
			echo "## Diff Stat (staged)"; \
			echo "${diff_cached_stat}"; \
			echo ""; \
			echo "## Risk Scan"; \
			if [ -n "${risk_scan}" ]; then echo "${risk_scan}"; else echo "No matches."; fi; \
			echo ""; \
			echo "## Cargo Check (tail)"; \
			printf '%s\n' '```'; \
			echo "${check_tail}"; \
			printf '%s\n' '```'; \
		} > "{{SYNC_FILE}}"; \
	fi; \
	{{CLIP}} < "{{SYNC_FILE}}"; \
	{{OPEN}} "{{CHAT_URL}}"; \
	echo "Copied sync_report.md to clipboard. Paste into ChatGPT."

drive:
	@{{PASTE}} > "{{PROMPT_FILE}}"
	@sed -n "1,200p" "{{PROMPT_FILE}}"
	@if [ -n "$${AGENT_CMD:-}" ]; then \
		eval "$${AGENT_CMD} \"$$(cat \"{{PROMPT_FILE}}\")\""; \
	else \
		echo "Prompt saved to .driver_prompt.md. Run your agent manually."; \
	fi

clean:
	@rm -f "{{CONTEXT_FILE}}" "{{DIFF_FILE}}" "{{SYNC_FILE}}" "{{PROMPT_FILE}}" context-*.xml diff-*.patch
