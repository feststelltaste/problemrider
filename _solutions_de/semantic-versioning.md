---
title: Semantic Versioning
description: Kommunikation der Kompatibilitätsabsicht durch strukturierte
  Versionsnummern.
category:
- Dependencies
- Process
problems:
- api-versioning-conflicts
- breaking-changes
- dependency-version-conflicts
- legacy-api-versioning-nightmare
- integration-difficulties
- ripple-effect-of-changes
- abi-compatibility-issues
- rapid-system-changes
layout: solution
lang: de
en_slug: semantic-versioning
related_solutions:
- slug: versioning-scheme
  similarity: 0.85
- slug: version-control
  similarity: 0.75
- slug: schema-registry
  similarity: 0.7
- slug: backward-compatibility
  similarity: 0.7
- slug: compatibility-as-error
  similarity: 0.7
- slug: compatibility-standards
  similarity: 0.7
---

## Description

Semantic Versioning kodiert Kompatibilitätsabsicht direkt in eine Versionsnummer — MAJOR für Breaking Changes, MINOR für rückwärtskompatible Erweiterungen, PATCH für rückwärtskompatible Fehlerbehebungen —, sodass Konsumenten einer Bibliothek oder API allein durch das Lesen der Version beurteilen können, ob ein Update sicher zu übernehmen ist, ohne ein Änderungsprotokoll oder den Code selbst zu inspizieren. Legacy-Ökosysteme, die aus vielen voneinander abhängigen gemeinsam genutzten Komponenten aufgebaut sind, sind besonders anfällig für unversionierte oder inkonsistent versionierte Änderungen, bei denen jedes Update einen nachgelagerten Konsumenten still brechen kann, ganz ohne Warnung. Die Nachrüstung von Semver auf Legacy-Komponenten erfordert ein anfängliches Audit, um eine solide Basisversion zu etablieren, aber einmal konsistent übernommen und über die CI-Pipeline durchgesetzt, gibt es jedem Team ein gemeinsames, zuverlässiges Vokabular für die Auswirkung einer Änderung und lässt automatisierte Abhängigkeitswerkzeuge sichere Upgrade-Entscheidungen selbstständig treffen.

## How to Apply ◆

- Übernehmen Sie das MAJOR.MINOR.PATCH-Versionierungsschema für alle Bibliotheken, APIs und gemeinsam genutzten Komponenten im Legacy-System.
- Definieren Sie klare Regeln: Erhöhen Sie MAJOR für Breaking Changes, MINOR für rückwärtskompatible Erweiterungen, PATCH für rückwärtskompatible Fehlerbehebungen.
- Rüsten Sie bestehende Legacy-Komponenten mit semantischen Versionen nach, indem Sie deren aktuelle Schnittstellen prüfen und eine Basisversion etablieren.
- Integrieren Sie Versionserhöhungen in die CI/CD-Pipeline, sodass Entwickler die Art der von ihnen vorgenommenen Änderung deklarieren müssen.
- Veröffentlichen Sie Änderungsprotokolle zusammen mit Versionserhöhungen, um zu dokumentieren, was sich geändert hat und warum.
- Nutzen Sie Abhängigkeitsverwaltungswerkzeuge, die semantische Versionsbereiche respektieren, um versehentliche Upgrades auf inkompatible Versionen zu verhindern.

## Tradeoffs ⇄

**Vorteile:**
- Konsumenten können durch Prüfung der Versionsnummer sofort verstehen, ob ein Update sicher zu übernehmen ist.
- Reduziert überraschende Brüche beim Upgrade von Abhängigkeiten in einem Legacy-Ökosystem.
- Bietet ein gemeinsames Vokabular über Teams hinweg für die Diskussion der Auswirkung von Änderungen.
- Ermöglicht automatisierten Abhängigkeits-Update-Werkzeugen, sichere Upgrade-Entscheidungen zu treffen.

**Kosten:**
- Erfordert Disziplin; falsche Versionserhöhungen untergraben das Vertrauen in das Schema.
- Legacy-Komponenten ohne vorherige Versionierung brauchen ein potenziell zeitaufwändiges Audit, um eine Basislinie zu etablieren.
- Verhindert keine Breaking Changes, kommuniziert sie nur; Durchsetzung erfordert zusätzliches Tooling.
- Pre-1.0-Versionen haben schwächere Garantien, was während früher Modernisierungsphasen zu Verwirrung führen kann.

## How It Could Be

Ein Legacy-Monolith wird in gemeinsam genutzte Bibliotheken zerlegt, die von mehreren Teams konsumiert werden. Ohne Semantic Versioning ziehen Teams häufig Updates ein, die still ihre Builds brechen. Nach der Übernahme von Semver veröffentlicht jede Bibliothek ein Änderungsprotokoll und erhöht ihre Major-Version, wenn sich APIs ändern. Konsumierende Teams konfigurieren ihre Abhängigkeitsverwalter so, dass sie nur kompatible Minor- und Patch-Updates automatisch akzeptieren. Major-Versionserhöhungen lösen eine geplante Migrationsanstrengung aus. Über sechs Monate sinken ungeplante Brüche durch Bibliotheks-Updates erheblich, und Teams gewinnen Vertrauen beim Upgrade von Abhängigkeiten, weil die Versionsnummer die Absicht zuverlässig signalisiert.
