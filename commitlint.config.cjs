// Commitlint configuration.
//
// Extends config-conventional (Angular) — see CLAUDE.md for the
// allowed type list. The only deviation: Dependabot's auto-generated
// commit subjects start with "Bump <pkg> from X to Y", which fails the
// default `subject-case` rule (start-case). Exempt those.

module.exports = {
  extends: ['@commitlint/config-conventional'],
  ignores: [
    // Match Dependabot's standard "Bump" subject (case-insensitive on
    // the verb so a future Dependabot tweak doesn't slip through). The
    // strict commitlint rules still apply to every other commit,
    // including manual `chore(deps):` updates that don't follow this
    // pattern.
    (message) => /^chore\(deps(-dev)?\):\s+[Bb]ump\s+/.test(message),
    // release-please squash merges carry an auto-generated
    // "Co-authored-by: <bot>[bot] <id+login@users.noreply...>" trailer
    // that exceeds body-max-line-length (100). The commit is entirely
    // machine-generated; exempt it.
    (message) => /^chore\(main\): release /.test(message),
  ],
};
