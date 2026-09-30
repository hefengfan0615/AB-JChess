# Contributing to JieqiBox

Thank you for your interest in contributing to JieqiBox! This document provides guidelines and information for contributors.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Set up the development environment**:
   ```bash
   npm install
   cd src-tauri && cargo build && cd ..
   ```

## Development Workflow

### Code Style

- **Vue Components**: Use `<script setup>` syntax with TypeScript
- **CSS**: Use SCSS with scoped styles
- **Rust**: Follow Rust coding conventions
- **TypeScript**: Use strict mode and proper typing

### Commit Messages

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test changes
- `chore`: Build/tooling changes

### Pull Request Process

1. **Create a feature branch** from `main`
2. **Make your changes** following the coding standards
3. **Test your changes** thoroughly
4. **Update documentation** if needed
5. **Submit a pull request** with a clear description

### Testing

- Run the development server: `npm run dev`
- Test the Tauri app: `npm run tauri dev`
- Build for production: `npm run tauri build`

## Project Structure

```
jieqibox/
├── src/                    # Vue frontend
│   ├── components/         # Vue components
│   ├── composables/        # Vue composables
│   └── assets/            # Static assets
├── src-tauri/             # Rust backend
│   ├── src/               # Rust source
│   └── Cargo.toml         # Rust dependencies
└── public/                # Public assets
```

## Reporting Issues

When reporting issues, please include:

1. **Operating system** and version
2. **Node.js version**
3. **Rust version**
4. **Steps to reproduce**
5. **Expected vs actual behavior**
6. **Screenshots** if applicable

## Feature Requests

For feature requests:

1. Check existing issues first
2. Create a new issue with the `enhancement` label
3. Describe the feature clearly
4. Explain the use case and benefits

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Follow the project's coding standards

## License

By contributing to JieqiBox, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to JieqiBox! 🎉
