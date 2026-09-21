---
title: Sicherheitsrichtlinien für die Entwicklung
description: Definition verbindlicher Regeln für die sichere
  Softwareentwicklung.
category:
- Security
- Process
problems:
- inconsistent-coding-standards
- undefined-code-style-guidelines
- process-design-flaws
- inadequate-code-reviews
- inconsistent-quality
- poor-documentation
layout: solution
lang: de
en_slug: security-policies-for-development
related_solutions:
- slug: secure-software-development
  similarity: 0.85
- slug: secure-coding-guidelines
  similarity: 0.8
- slug: security-policies-for-users
  similarity: 0.8
- slug: security-training
  similarity: 0.7
- slug: security-tests
  similarity: 0.7
- slug: security-culture
  similarity: 0.7
---

## Description

Sicherheitsrichtlinien für die Entwicklung sind verbindliche, dokumentierte Regeln, die regeln, wie Software gebaut wird — abdeckend sichere Programmierpraktiken, Code-Review-Anforderungen, Abhängigkeitsmanagement und die Behandlung von Secrets —, die eine konsistente Basiserwartung über Teams hinweg etablieren, statt Sicherheitspraxis dem individuellen Wissen und Urteilsvermögen jedes Entwicklers zu überlassen. Der Mechanismus ersetzt explizite Regeln und automatisierte Durchsetzung, wie Pre-Commit-Hooks und CI-Pipeline-Prüfungen, für stillschweigende Konvention, sodass es nicht mehr davon abhängt, welche Entwickler zufällig in diesem Team sind, ob ein gegebenes Team Secrets in Versionskontrolle committet, Eingaben konsistent validiert oder sicherheitssensible Codepfade überprüft. Dies zählt besonders dort, wo eine Organisation viele parallele Teams betreibt, die separate Legacy-Codebasen betreuen, weil ohne gemeinsame Richtlinie die Praktiken jedes Teams unabhängig über Jahre auseinanderdriften, typischerweise zu inkonsistenten und manchmal widersprüchlichen Konventionen für genau dasselbe Anliegen konvergierend — wie wenn manche Teams Umgebungsvariablen für Anmeldedaten nutzen, während andere sie fest codieren und wieder andere gar keinen konsistenten Ansatz haben. Eine schriftliche Richtlinie allein ändert wenig; ihr Effekt kommt daher, mit automatisierter Durchsetzung gepaart zu werden, die Verstöße am Änderungspunkt sichtbar und blockiert macht statt später entdeckt zu werden, und davon, so kalibriert zu sein, dass der bestehende Rückstand an Verstößen einer Legacy-Codebasis auf pragmatischem Zeitplan behoben wird, statt am ersten Tag undurchführbare pauschale Durchsetzung auszulösen. Für Modernisierungsaufwände ist der Wert dieser Lösung die Etablierung einer dauerhaften Basislinie, die verhindert, dass neu eingeführter Code dieselben Inkonsistenzen reproduziert, von denen sich der Modernisierungsaufwand entfernen will.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Definieren Sie Richtlinien, die sichere Programmierpraktiken, Code-Review-Anforderungen, Abhängigkeitsmanagement und Secret-Behandlung abdecken
- Etablieren Sie verpflichtende Sicherheitsprüfungen an Schlüsselgates des Entwicklungslebenszyklus wie Design-Review, Code-Merge und Veröffentlichung
- Verlangen Sie, dass alle Codeänderungen automatisierte Sicherheitsscans bestehen, bevor sie gemergt werden
- Verlangen Sie Peer-Review für sicherheitssensible Codepfade, einschließlich Authentifizierung, Autorisierung und Datenbehandlung
- Definieren Sie akzeptable und verbotene Praktiken für die Behandlung sensibler Daten in Code, Logs und Konfiguration
- Setzen Sie Branch-Protection-Regeln durch, die die Umgehung von Sicherheitsrichtlinienanforderungen verhindern
- Überprüfen und aktualisieren Sie Richtlinien jährlich oder wenn bedeutende neue Bedrohungen oder Technologien eingeführt werden

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Schafft konsistente Sicherheitserwartungen über alle Entwicklungsteams hinweg
- Reduziert die Abhängigkeit vom individuellen Sicherheitswissen der Entwickler
- Liefert klare Leitlinien, die sicherheitsbezogene Entscheidungsfindung vereinfachen
- Unterstützt Audit und Compliance durch Dokumentation verbindlicher Sicherheitspraktiken

**Kosten und Risiken:**
- Zu restriktive Richtlinien können die Entwicklungsgeschwindigkeit verlangsamen und Teams frustrieren
- Ohne Durchsetzungsmechanismen werden Richtlinien zu ambitionierten Dokumenten, die ignoriert werden
- Legacy-Codebasen könnten umfangreiche Richtlinienverstöße haben, die pragmatische Behebungszeitpläne erfordern
- Die Richtlinienpflege erfordert laufende Aufmerksamkeit, um relevant und effektiv zu bleiben

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Softwareunternehmen mit 15 Entwicklungsteams fand, dass jedes Team unterschiedliche Standards für die Behandlung von API-Schlüsseln, Passwörtern und Tokens in seinen Legacy-Codebasen hatte. Manche Teams committeten Secrets in Versionskontrolle, andere nutzten Umgebungsvariablen inkonsistent, und einige hatten überhaupt keine Richtlinie. Das Sicherheitsteam verfasste eine prägnante Entwicklungssicherheitsrichtlinie, die Secret-Management, Input-Validierung, Logging-Beschränkungen und Abhängigkeits-Update-Anforderungen abdeckte. Sie automatisierten die Durchsetzung durch Pre-Commit-Hooks und CI-Pipeline-Prüfungen. Innerhalb von drei Monaten fielen Secret-im-Code-Befunde auf null, und alle Teams folgten denselben Basissicherheitspraktiken.
