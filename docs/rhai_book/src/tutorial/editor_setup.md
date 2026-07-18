# Editor Setup (LSP)

Conchordal ships a Rhai LSP definition file describing its named callable
scripting API, including hover documentation. With a compatible client, it can
provide completion, hover, go-to-definition, and diagnostics for calls such as
`place`, `harmonic`, `.brain()`, and `.send(habitat_bus)`. The bus-combining
`|` overloads are operators rather than named declarations and are therefore
not emitted into the definition file.

The two files that drive this are committed at the repo root:

- `Rhai.toml` — workspace config picked up by `rhai-lsp`
- `rhai-defs/conchordal.d.rhai` — auto-generated type/fn declarations with
  doc comments

The upstream [rhai-lsp](https://github.com/rhaiscript/lsp) project labels
itself experimental and incomplete, and does not recommend general use. Treat
this setup as an optional development aid rather than guaranteed production
tooling. If you still want to use it, install it directly from its git
repository (it is not published on crates.io):

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

## Neovim 0.11+

```lua
vim.lsp.config("rhai", {
  cmd = { "rhai", "lsp", "stdio", "--config", "Rhai.toml" },
  filetypes = { "rhai" },
  root_markers = { "Rhai.toml", ".git" },
})
vim.lsp.enable("rhai")
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
args = ["lsp", "stdio", "--config", "Rhai.toml"]
```

## Emacs (eglot)

```elisp
(add-to-list 'eglot-server-programs
             '(rhai-mode . ("rhai" "lsp" "stdio" "--config" "Rhai.toml")))
(add-hook 'rhai-mode-hook #'eglot-ensure)
```

## Regenerating the definition file

The definition file and both the English and Japanese API references are
generated from the engine's `register_fn` surface joined with the
documentation registry (`src/scripting/docs.rs`). If you pull a new conchordal
version and miss diagnostics, regenerate all three:

```bash
cargo run --bin gen_rhai_defs
```

CI tests fail whenever the committed artifacts are stale. This checks the
registered signatures and generated text; editor behavior still depends on
the experimental upstream server and the client configuration.
