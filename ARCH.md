# Architecture Documentation (arc42)

_Hint: AI generated; needs to be reviewed; but pretty good so far._

## 1. Introduction and Goals

Legacy system modernization is notoriously difficult because many problems are interconnected in complex ways that are hard to understand. Traditional approaches rely on subjective expert judgment to identify root causes and symptoms, leading to inconsistent analysis, missed dependencies and biases. Problem Rider addresses this by providing the first systematic, AI-validated catalog of legacy system problems with quantified causal relationships. The project consists of a comprehensive catalog of legacy system problems including trustworthy cause-effect relationships between problems. It provides software maintainers and architects with evidence-based insights for legacy system modernization endeavours. This enables data-driven modernization planning where teams can prioritize high-impact root causes and anticipate cascading effects with measurable confidence rather than pure guesswork.

### 1.1 Requirements Overview

The Problem Rider system is a knowledge catalog for documenting and organizing typical problems found in legacy systems. The catalog serves software maintainers, architects, and developers who need to:

- Analyze and understand legacy system problems
- Find solutions and patterns for common issues
- Navigate interconnected problem relationships
- Plan modernization strategies

### 1.2 Quality Goals

| Priority | Quality Goal | Description |
|----------|--------------|-------------|
| 1 | **Usability** | Easy navigation and problem discovery through categorization and linking |
| 2 | **Maintainability** | Simple content management with markdown files and automated tooling |
| 3 | **Scalability** | Support for growing catalog with hundreds of problems |
| 4 | **Accessibility** | Static site accessible via GitHub Pages with responsive design |

### 1.3 Stakeholders

| Role | Expectations |
|------|-------------|
| Software Architects | Comprehensive problem patterns for legacy modernization |
| Maintainers | Easy content updates and link management |
| Developers | Quick problem identification and solution guidance |
| Contributors | Simple workflow for adding new problems |

## 2. Architecture Constraints

### 2.1 Technical Constraints

- **Static Site Generation**: Jekyll-based for GitHub Pages deployment
- **Content Format**: Markdown files with YAML front matter
- **Version Control**: Git-based content management
- **Hosting**: GitHub Pages (free, reliable, integrated with repository)

### 2.2 Organizational Constraints

- **Open Source**: Public repository with contribution guidelines
- **No Backend**: Static site only, no server-side processing
- **Minimal Dependencies**: Standard Jekyll setup for broad compatibility

## 3. System Scope and Context

### 3.1 Business Context

```
[Contributors] --write--> [Problem Catalog] --serves--> [Software Teams]
     ^                           |                          |
     |                           v                          |
[Maintainers] <--manage-- [GitHub Repository] <--feedback--+
```

### 3.2 Technical Context

```
[Markdown Files] --processed--> [Jekyll] --generates--> [Static Site]
      |                           |                         |
      v                           v                         v
[Git Repository] <--triggers-- [GitHub Actions] --deploys--> [GitHub Pages]
```

## 4. Solution Strategy

### 4.1 Technology Decisions

- **Jekyll**: Static site generator for simple deployment and maintenance
- **Markdown**: Human-readable format for content creation
- **YAML Front Matter**: Structured metadata for categorization and linking
- **GitHub Pages**: Hosting solution with CI/CD integration
- **Python Scripts**: Automation for maintenance tasks and causal analysis

### 4.2 Content Organization

- **Problem-Centric**: Each problem is a separate markdown file
- **Category-Based**: 15 core categories for logical grouping
- **Link-Rich**: Extensive cross-referencing between related problems
- **Metadata-Driven**: YAML front matter enables automated features

## 5. Building Block View

### 5.1 Level 1: System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Problem Rider System                     │
├─────────────────┬─────────────────┬─────────────────────────┤
│   Content       │   Processing    │      Presentation       │
│   Management    │   Layer         │      Layer              │
│                 │                 │                         │
│ ┌─────────────┐ │ ┌─────────────┐ │ ┌─────────────────────┐ │
│ │ Markdown    │ │ │ Jekyll      │ │ │ Static Website      │ │
│ │ Files       │→│ │ Generator   │→│ │ - Problem Pages     │ │
│ │             │ │ │             │ │ │ - Category Pages    │ │
│ └─────────────┘ │ └─────────────┘ │ │ - Navigation        │ │
│                 │                 │ └─────────────────────┘ │
│ ┌─────────────┐ │ ┌─────────────┐ │                         │
│ │ Python      │ │ │ YAML        │ │                         │
│ │ Scripts     │ │ │ Processing  │ │                         │
│ │             │ │ │             │ │                         │
│ └─────────────┘ │ └─────────────┘ │                         │
└─────────────────┴─────────────────┴─────────────────────────┘
```

### 5.2 Level 2: Content Structure

```
_problems/
├── problem-1.md (YAML front matter + Markdown content)
├── problem-2.md
└── ...

Each problem file contains:
- title: Problem name
- description: Brief summary
- category: [Category1, Category2, ...]
- related_problems: [slug1, slug2, ...]
- Content sections: Description, Symptoms, Root Causes, etc.
```

## 6. Runtime View

### 6.1 Content Publication Flow

```
1. Author generates problem.md according to the problem pattern
2. Author connects problem with others
   - checks symptoms/causes manually or via scripting
   - recalculates related using scripts
   - condenses categories to have small set of categories to work with
3. Git commit/push triggers GitHub Actions
4. Jekyll processes markdown files
5. Static site generation:
   - Problem pages created from templates
   - Category pages generated with listings
   - Cross-references resolved
5. Site deployed to GitHub Pages
6. Users access via web browser
```

## 7. Deployment View

### 7.1 Infrastructure

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   GitHub        │    │   GitHub        │    │   GitHub        │
│   Repository    │───▶│   Actions       │───▶│   Pages         │
│                 │    │   (CI/CD)       │    │   (Hosting)     │
│ - Source files  │    │ - Jekyll build  │    │ - Static site   │
│ - Scripts       │    │ - Deploy        │    │ - CDN delivery  │
│ - Templates     │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 7.2 Build Process

1. **Trigger**: Git push to main branch
2. **Environment**: Ubuntu runner with Ruby/Jekyll
3. **Build Steps**:
   - Install dependencies (bundle install)
   - Generate site (jekyll build)
   - Deploy to gh-pages branch
4. **Deployment**: Automatic to GitHub Pages CDN

## 8. Cross-Cutting Concepts

### 8.1 Category System

- **15 Core Categories**: Process, Architecture, Code, Performance, Team, Communication, Management, Security, Business, Operations, Testing, Database, Dependencies, Requirements, Culture
- **Consolidation**: Script-based migration from 200+ to 15 categories
- **Linking**: Categories link to filtered problem listings

### 8.2 Problem Relationships

- **Related Problems**: YAML-based linking with semantic similarity scores  
- **Cross-References**: Markdown links within content
- **Bidirectional**: Scripts maintain link consistency
- **Causal Analysis**: Symptoms and causes are authored, then checked structurally by `validate_causal_links.py`


## 9. Architecture Decisions

### 9.1 Static Site Generation

**Decision**: Use Jekyll for static site generation

**Rationale**:
- No server maintenance required
- Fast loading and high availability
- Version control for all content
- GitHub Pages integration
- Security through simplicity

**Consequences**:
- Limited dynamic features
- Build-time processing only
- Git-based workflow required

### 9.2 Markdown Content Format

**Decision**: Use Markdown with YAML front matter

**Rationale**:
- Human-readable and writable
- Version control friendly
- Tool ecosystem support
- Structured metadata capability

**Consequences**:
- Learning curve for non-technical contributors
- Processing required for advanced features
- Template consistency important

### 9.3 Structural Validation of Causal Claims

**Decision**: Author symptoms and causes, and validate them structurally rather than
generating them from a causal scoring model

**Rationale**:
- An automated causal scoring model (CESAR) was trialled and abandoned: it produced
  near-symmetric scores for unrelated problem pairs, so it could not decide direction
- Structural checks need no model and catch the defects that actually occur:
  a claim recorded on only one of the two problem files, a problem asserted as both
  cause and effect, and causal cycles
- The asymmetry rate is the catalog's main quality measure for these links, since
  every claim should appear in both the causing and the affected problem

**Consequences**:
- Dependency on external AI model and Python environment
- Cache management required for incremental processing
- Human oversight needed to validate AI-generated relationships


## 10. Quality Requirements

### 10.1 Performance

- **Page Load**: < 2 seconds for problem pages
- **Build Time**: < 5 minutes for full site generation
- **Navigation**: Instant category jumping via anchor links

### 10.2 Maintainability

- **Content Updates**: Simple markdown editing
- **Link Management**: Automated checking and fixing
- **Category Management**: Script-based consolidation
- **Related Problems**: Automated similarity calculation

### 10.3 Usability

- **Problem Discovery**: Multiple navigation paths (categories, search, links)
- **Mobile Responsive**: Works on all device sizes
- **Accessibility**: Proper semantic HTML and ARIA labels

## 11. Risks and Technical Debt

### 11.1 Technical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| GitHub Pages limitations | Medium | Document workarounds, consider alternatives |
| Build failures | High | Comprehensive testing, backup deployment |
| Link rot | Medium | Automated link checking scripts |
| Category proliferation | Low | Strict guidelines, consolidation scripts |

### 11.2 Current Technical Debt

- **Limited search**: No full-text search capability  
- **Mobile optimization**: Some components need responsive improvements

## 12. Glossary

| Term | Definition |
|------|------------|
| **Problem** | A documented issue pattern with symptoms, causes, and examples |
| **Category** | One of 15 core classification groups for problems |
| **Related Problems** | Cross-referenced problems with similarity scores |
| **Slug** | URL-friendly identifier derived from problem title |
| **Front Matter** | YAML metadata at the top of markdown files |
| **Jekyll** | Ruby-based static site generator |
