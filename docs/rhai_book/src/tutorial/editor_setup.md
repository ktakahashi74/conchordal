# Editor Setup (LSP)

Conchordal ships a Rhai LSP definition file describing the entire scripting
surface, including hover documentation for every function. Hooking your editor
up to it gives you completion, hover, go-to-def, and inline diagnostics for
every conchordal function — `place`, `harmonic`, `.brain()`,
`.send(habitat_bus)`, and so on.

The two files that drive this are committed at the repo root:

- `Rhai.toml` — workspace config picked up by `rhai-lsp`
- `rhai-defs/conchordal.d.rhai` — auto-generated type/fn declarations with
  doc comments

Install the [rhai-lsp](https://github.com/rhaiscript/lsp) server once (it is
not on crates.io; install directly from the git repo):

```bash
cargo install --git https://github.com/rhaiscript/lsp rhai-cli
```

This builds a binary named `rhai` with the `lsp` subcommand.

Then wire your editor:

## VS Code

The official [Rhai](https://marketplace.visualstudio.com/items?itemName=rhaiscript.vscode-rhai)
extension currently provides syntax highlighting only. It does not launch
`rhai-lsp`. For LSP features in VS Code, use an LSP client/extension that can
launch this command from the conchordal workspace:

```bash
rhai lsp stdio --config Rhai.toml
```

## Neovim (nvim-lspconfig)

```lua
require("lspconfig").rhai.setup({
  cmd = { "rhai", "lsp", "stdio" },
  filetypes = { "rhai" },
  root_dir = require("lspconfig.util").root_pattern("Rhai.toml", ".git"),
})
```

## Helix

In `~/.config/helix/languages.toml`:

```toml
[[language]]
name = "rhai"
scope = "source.rhai"
file-types = ["rhai"]
language-servers = ["rhai-lsp"]

[language-server.rhai-lsp]
command = "rhai"
args = ["lsp", "stdio"]
```

## Emacs (eglot)

```elisp
(add-to-list 'eglot-server-programs
             '(rhai-mode . ("rhai" "lsp" "stdio")))
(add-hook 'rhai-mode-hook #'eglot-ensure)
```

## Regenerating the definition file

The definition file and the [API Reference](../reference/api.md) are both
generated from the engine's `register_fn` surface joined with the
documentation registry (`src/scripting/docs.rs`). If you pull a new conchordal
version and miss diagnostics, regenerate both:

```bash
cargo run --bin gen_rhai_defs
```

CI tests fail whenever the committed artifacts are stale, so in a clean
checkout they are always in sync with the engine.
