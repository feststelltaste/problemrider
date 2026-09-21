---
title: Sicherheitstests
description: Verifikation von Sicherheitseigenschaften durch spezialisierte
  Testmethoden.
category:
- Security
- Testing
problems:
- insufficient-testing
- poor-test-coverage
- legacy-code-without-tests
- sql-injection-vulnerabilities
- cross-site-scripting-vulnerabilities
- authentication-bypass-vulnerabilities
- high-defect-rate-in-production
- session-management-issues
layout: solution
lang: de
en_slug: security-tests
related_solutions:
- slug: regression-tests
  similarity: 0.85
- slug: secure-software-development
  similarity: 0.85
- slug: static-code-analysis
  similarity: 0.85
- slug: secure-coding-guidelines
  similarity: 0.8
- slug: security-tests-by-external-parties
  similarity: 0.8
- slug: vulnerability-scans
  similarity: 0.8
---

## Description

Sicherheitstests verifizieren spezifische Sicherheitseigenschaften — Authentifizierung, Autorisierung, Input-Validierung, kryptografische Korrektheit — durch spezialisierte Methoden wie statische Analyse (SAST), dynamisches Scanning (DAST) und gezielte Unit-Tests, statt Sicherheit als etwas zu behandeln, das nur gelegentlich durch ein separates Audit geprüft wird. Die Integration dieser Tests in die CI/CD-Pipeline erfasst Schwachstellen in dem Moment, in dem sie eingeführt werden, was für aktiv während der Modernisierung modifizierte Legacy-Codebasen enorm zählt, da jedes Refactoring eine Gelegenheit ist, ein Schwachstellenmuster wieder einzuführen, für das der Code bereits einmal behoben wurde. Automatisierte Sicherheitstests bringen unvermeidlich falsch positive Ergebnisse hervor, die Expertentriage erfordern, um sie von echten Befunden zu unterscheiden, und sie verifizieren bekannte Schwachstellenmuster, statt die Abwesenheit neuartiger Angriffe zu garantieren, aber das wiederholbare Sicherheitsnetz, das sie bieten, ist es, was es überhaupt möglich macht, sicherheitssensiblen Legacy-Code mit irgendeiner Zuversicht anzufassen.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Implementieren Sie Static Application Security Testing (SAST), um Quellcode auf Schwachstellenmuster zu scannen
- Setzen Sie Dynamic Application Security Testing (DAST) ein, um laufende Anwendungen auf ausnutzbare Schwächen zu prüfen
- Fügen Sie Interactive Application Security Testing (IAST) für Laufzeitanalyse während der funktionalen Testausführung hinzu
- Erstellen Sie sicherheitsfokussierte Unit-Tests für Authentifizierung, Autorisierung, Input-Validierung und kryptografische Funktionen
- Integrieren Sie Sicherheitstests in die CI/CD-Pipeline, um Schwachstellen vor der Bereitstellung zu erfassen
- Pflegen Sie eine Bibliothek von Sicherheitstestfällen basierend auf OWASP Top 10 und Befunden aus vergangenen Vorfällen
- Planen Sie periodische umfassende Sicherheitstestläufe über das hinaus, was die CI-Pipeline abdeckt

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Erfasst Sicherheitsschwachstellen, bevor sie Produktionsumgebungen erreichen
- Bietet wiederholbare, automatisierte Verifikation von Sicherheitseigenschaften
- Baut Entwicklerbewusstsein für Sicherheitsprobleme durch unmittelbares Feedback auf
- Schafft ein Sicherheitsnetz während Legacy-Code-Refactoring und -Modernisierung

**Kosten und Risiken:**
- Sicherheitstestwerkzeuge produzieren falsch positive Ergebnisse, die Expertentriage erfordern
- Legacy-Codebasen könnten schwer für dynamisches Testing zu instrumentieren sein
- Sicherheitstests fügen der Build-Pipeline Ausführungszeit hinzu
- Werkzeuglizenzen und -pflege stellen laufende Kosten dar
- Tests verifizieren bekannte Schwachstellenmuster, können aber die Abwesenheit neuartiger Angriffe nicht garantieren

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Einzelhandelsunternehmen integrierte OWASP ZAP in seine CI-Pipeline für seine Legacy-E-Commerce-Anwendung. Während des ersten vollständigen Scans identifizierte das Werkzeug 23 potenzielle Schwachstellen, einschließlich reflektiertem XSS in der Suchfunktion, fehlenden Sicherheits-Headern und Informationsoffenlegung durch ausführliche Fehlermeldungen. Nach der Triage falsch positiver Ergebnisse bestätigte das Team 15 echte Probleme und behob sie über zwei Sprints. Die automatisierten Sicherheitstests verhinderten dann, dass drei ähnliche Schwachstellen während nachfolgender Entwicklung wieder eingeführt wurden, jede erfasst auf der Pull-Request-Stufe, bevor sie den Hauptbranch erreichten.
