---
title: Richtlinien für sichere Programmierung
description: Definition verbindlicher Regeln und bewährter Praktiken für
  sichere Programmierung.
category:
- Security
- Code
problems:
- inconsistent-coding-standards
- undefined-code-style-guidelines
- sql-injection-vulnerabilities
- cross-site-scripting-vulnerabilities
- buffer-overflow-vulnerabilities
- inadequate-code-reviews
- lower-code-quality
- inadequate-error-handling
- log-injection-vulnerabilities
layout: solution
lang: de
en_slug: secure-coding-guidelines
related_solutions:
- slug: secure-software-development
  similarity: 0.85
- slug: security-tests
  similarity: 0.8
- slug: security-training
  similarity: 0.8
- slug: static-code-analysis
  similarity: 0.8
- slug: security-policies-for-development
  similarity: 0.8
- slug: secure-programming-interfaces
  similarity: 0.8
---

## Description

Richtlinien für sichere Programmierung sind eine schriftliche, verbindliche Regelmenge, die festlegt, wie Entwickler wiederkehrende Risikobereiche wie Input-Validierung, Output-Encoding, Authentifizierung, Session-Management und Fehlerbehandlung handhaben müssen, zugeschnitten auf die spezifischen Sprachen und Frameworks, die das System tatsächlich nutzt. Ihr Wert hängt davon ab, durchgesetzt statt nur veröffentlicht zu werden — durch automatisierte statische Analyse, integriert in CI-Builds, und durch explizite Checklisten-Punkte im Code-Review —, sodass die Richtlinien formen, was gemergt wird, statt als Dokument zu existieren, das niemand konsultiert. Legacy-Codebasen sind häufig das Produkt jahrelanger Anhäufung durch verschiedene Teams und sogar verschiedene Generationen von Entwicklern, jede mit ihren eigenen, oft inkonsistenten Annahmen darüber, wie Eingaben zu validieren oder Fehler sicher zu behandeln sind, was bedeutet, dass dieselbe Schwachstellenklasse in Dutzenden unabhängig geschriebenen Variationen über den Code hinweg auftauchen kann. Richtlinien für sichere Programmierung direkt aus einer Analyse der tatsächlich in dieser Codebasis gefundenen Schwachstellenmuster zu verfassen — statt eine generische Branchen-Checkliste zu übernehmen — macht die Richtlinien sofort relevant und erlaubt es, statische Analyseregeln so abzustimmen, dass sie genau die Fehler erfassen, die das Team historisch gemacht hat. Die Richtlinien reduzieren außerdem die Abhängigkeit des Systems von der Sicherheitsexpertise einzelner Entwickler, da der Standard selbst, statt institutionellen Gedächtnisses, zur Basislinie wird, an der sowohl neue als auch erfahrene Entwickler gemessen werden.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Etablieren Sie eine schriftliche Menge sicherer Programmierstandards, zugeschnitten auf die im Legacy-System genutzten Sprachen und Frameworks
- Beziehen Sie Regeln für Input-Validierung, Output-Encoding, Authentifizierung, Session-Management und Fehlerbehandlung ein
- Integrieren Sie automatisierte statische Analysewerkzeuge, die die Richtlinien während CI-Builds durchsetzen
- Verlangen Sie Compliance mit den Richtlinien für sichere Programmierung als Teil der Code-Review-Checklisten
- Bieten Sie Schulungssitzungen an, die Entwickler durch häufige, in der bestehenden Codebasis gefundene Schwachstellenmuster führen
- Pflegen Sie ein lebendes Dokument, das sich mit neuen Bedrohungsentdeckungen und Technologieänderungen weiterentwickelt
- Erstellen Sie Codebeispiele, die sowohl unsichere Legacy-Muster als auch ihre sicheren Ersetzungen zeigen

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Verhindert, dass häufige Schwachstellenklassen in die Codebasis eingeführt werden
- Schafft ein gemeinsames Sicherheitsvokabular und eine Basislinie über das Entwicklungsteam hinweg
- Reduziert die Abhängigkeit vom Sicherheitswissen einzelner Entwickler
- Macht Code-Reviews effektiver, indem objektive Kriterien für Sicherheitsbewertung geliefert werden

**Kosten und Risiken:**
- Richtlinien erfordern laufende Pflege und Aktualisierungen, während sich Bedrohungen weiterentwickeln
- Übermäßig präskriptive Regeln können die Entwicklung verlangsamen und erfahrene Entwickler frustrieren
- Compliance ohne Verständnis kann zu Cargo-Cult-Sicherheitspraktiken führen
- Legacy-Codebasen könnten umfangreiche Verstöße haben, die rückwirkend teuer zu beheben sind

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein schnell gewachsenes Fintech-Startup entdeckte, dass seine fünf Jahre alte Java-Codebasis inkonsistente Ansätze für Input-Validierung und Fehlerbehandlung über verschiedene Module hinweg enthielt, jedes von einem anderen Team geschrieben. Das Sicherheitsteam verfasste ein Dokument mit Richtlinien für sichere Programmierung, das die Top-15-Schwachstellenmuster abdeckte, die in ihrem Code gefunden wurden, zusammen mit genehmigten Behebungsmustern. Sie integrierten SonarQube-Regeln, die diesen Richtlinien entsprachen, in die Build-Pipeline. Innerhalb von vier Monaten fielen neue Code-Verstöße um 75 %, und die Richtlinien wurden zu einer zentralen Ressource während des Entwickler-Onboardings.
