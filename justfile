set shell := ["bash", "-euo", "pipefail", "-c"]
set dotenv-load := true

SESSION_FILE := ".chat_url"
PROMPT_FILE := ".driver_prompt.md"
SYNC_FILE := "sync_report.md"
CONTEXT_FILE := "context.xml"
DIFF_FILE := "diff.patch"
REQUEST_FILE := "chat_request.md"
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
	loc="$${LC_ALL:-$${LC_MESSAGES:-$${LANG:-}}}"; \
	if [[ "$${loc}" == ja* || "$${loc}" == *ja_JP* || "$${loc}" == *_JP* ]]; then \
		lang_line="Language: Please respond in Japanese."; \
		request_body="添付のdiff（および必要ならログ/エラー出力）を読み、実装が仕様どおりに完了しているかレビューしてください。不完全・不整合・潜在バグ・テスト不足があれば、Codexにそのまま渡せる“修正プロンプト”を箇条書きで生成してください（必要なら追加テスト/検証手順も含めて）。"; \
	else \
		lang_line="Language: Please respond in English."; \
		request_body="Review the attached diff (and logs/errors if present) and verify the implementation is complete and correct relative to the intended spec. If anything is incomplete, inconsistent, or risky (including missing tests), produce a Codex-ready fix prompt as a bullet list (include additional tests/verification steps when needed)."; \
	fi; \
	{ \
		echo "# Diff return note"; \
		echo ""; \
		echo "I'm uploading context.xml and diff.patch (or their timestamped equivalents) for review."; \
		echo ""; \
		echo "## Request"; \
		echo "$${lang_line}"; \
		echo "$${request_body}"; \
	} > "{{REQUEST_FILE}}"; \
	{{CLIP}} < "{{REQUEST_FILE}}"; \
	{{OPEN}} .; \
	{{OPEN}} "{{CHAT_URL}}"; \
	echo "Upload context-{{TS}}.xml and diff-{{TS}}.patch (recommended)."; \
	echo "context.xml and diff.patch are stable aliases of the latest run."; \
	echo "Paste chat_request.md from clipboard."

diff:
	@printf "### git diff (unstaged)\n" > "diff-{{TS}}.patch"; \
	git diff >> "diff-{{TS}}.patch"; \
	printf "\n### git diff --cached (staged)\n" >> "diff-{{TS}}.patch"; \
	git diff --cached >> "diff-{{TS}}.patch"; \
	ln -sf "diff-{{TS}}.patch" "{{DIFF_FILE}}"; \
	loc="$${LC_ALL:-$${LC_MESSAGES:-$${LANG:-}}}"; \
	if [[ "$${loc}" == ja* || "$${loc}" == *ja_JP* || "$${loc}" == *_JP* ]]; then \
		lang_line="Language: Please respond in Japanese."; \
		request_body="添付のdiff（および必要ならログ/エラー出力）を読み、実装が仕様どおりに完了しているかレビューしてください。不完全・不整合・潜在バグ・テスト不足があれば、Codexにそのまま渡せる“修正プロンプト”を箇条書きで生成してください（必要なら追加テスト/検証手順も含めて）。"; \
	else \
		lang_line="Language: Please respond in English."; \
		request_body="Review the attached diff (and logs/errors if present) and verify the implementation is complete and correct relative to the intended spec. If anything is incomplete, inconsistent, or risky (including missing tests), produce a Codex-ready fix prompt as a bullet list (include additional tests/verification steps when needed)."; \
	fi; \
	{ \
		echo "# Diff return note"; \
		echo ""; \
		echo "I'm uploading diff.patch for review."; \
		echo ""; \
		echo "## Request"; \
		echo "$${lang_line}"; \
		echo "$${request_body}"; \
	} > "{{REQUEST_FILE}}"; \
	{{CLIP}} < "{{REQUEST_FILE}}"; \
	{{OPEN}} "{{CHAT_URL}}"; \
	echo "Upload diff-{{TS}}.patch (recommended)."; \
	echo "diff.patch is a stable alias of the latest run."; \
	echo "Paste chat_request.md from clipboard."

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
	loc="$${LC_ALL:-$${LC_MESSAGES:-$${LANG:-}}}"; \
	if [[ "$${loc}" == ja* || "$${loc}" == *ja_JP* || "$${loc}" == *_JP* ]]; then \
		lang_line="Language: Please respond in Japanese."; \
		request_body="添付のdiff（および必要ならログ/エラー出力）を読み、実装が仕様どおりに完了しているかレビューしてください。不完全・不整合・潜在バグ・テスト不足があれば、Codexにそのまま渡せる“修正プロンプト”を箇条書きで生成してください（必要なら追加テスト/検証手順も含めて）。"; \
	else \
		lang_line="Language: Please respond in English."; \
		request_body="Review the attached diff (and logs/errors if present) and verify the implementation is complete and correct relative to the intended spec. If anything is incomplete, inconsistent, or risky (including missing tests), produce a Codex-ready fix prompt as a bullet list (include additional tests/verification steps when needed)."; \
	fi; \
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
	{ \
		echo ""; \
		echo "## Request"; \
		echo "$${lang_line}"; \
		echo "$${request_body}"; \
	} >> "{{SYNC_FILE}}"; \
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
	@rm -f "{{CONTEXT_FILE}}" "{{DIFF_FILE}}" "{{SYNC_FILE}}" "{{PROMPT_FILE}}" "{{REQUEST_FILE}}" context-*.xml diff-*.patch
