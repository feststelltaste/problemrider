---
title: Continuous Integration
description: Regelmäßige Integration von Codeänderungen in ein gemeinsames Repository.
category:
- Process
- Testing
problems:
- regression-bugs
- breaking-changes
- long-build-and-test-times
- merge-conflicts
- integration-difficulties
- long-lived-feature-branches
- deployment-risk
- high-bug-introduction-rate
- large-pull-requests
- reduced-code-submission-frequency
layout: solution
lang: de
en_slug: continuous-integration
related_solutions:
- slug: continuous-integration-and-delivery
  similarity: 0.9
- slug: integration-tests
  similarity: 0.85
- slug: trunk-based-development
  similarity: 0.8
- slug: compatibility-as-error
  similarity: 0.75
- slug: automated-tests
  similarity: 0.75
- slug: canary-releases
  similarity: 0.75
---

## Description

Continuous Integration verlangt von jedem Entwickler, Codeänderungen häufig — idealerweise mindestens täglich — in einen gemeinsamen Hauptbranch zu mergen, wobei bei jeder Integration ein automatisierter Build- und Testlauf ausgelöst wird, sodass Konflikte und Regressionen innerhalb von Minuten statt unbemerkt über langlebige Branches angehäuft erkannt werden. Legacy-Codebasen, denen diese Disziplin fehlt, entwickeln tendenziell Integrationszyklen, die in Wochen gemessen werden, wo Branches so lange divergieren, dass ihr Mergen zu einer eigenen, gefürchteten Aktivität wird, die Tage von Konfliktlösung und Regressionsjagd beinhaltet, was Entwickler wiederum davon abhält, häufiger zu integrieren, und das Muster verstärkt. Die Feedback-Schleife schnell zu machen — häufig als unter fünfzehn Minuten genannt — ist das, was häufige Integration praktikabel statt bloß vorgeschrieben macht, da eine langsame Pipeline denselben Anreiz zur Bündelung von Änderungen wiederherstellt, den langlebige Branches ursprünglich geschaffen haben. Dieser Pipeline Kompatibilitäts- und Vertragstests neben Unit-Tests hinzuzufügen erweitert ihren Wert über das Erkennen von Logikregressionen hinaus auf das automatische Erkennen schnittstellenbrechender Änderungen, was besonders in Legacy-Systemen wichtig ist, wo undokumentierte Abhängigkeiten zwischen Komponenten häufig sind. Die Effektivität der Praxis ist durch den Zustand der bestehenden Testsuite begrenzt: Eine Legacy-Codebasis mit wenig oder keiner Testabdeckung erhält von Continuous Integration ein Build-Signal, aber noch nicht das Sicherheitsnetz, das häufige Integration risikoarm macht, sodass Testinvestition und CI-Einführung tendenziell gemeinsam fortschreiten müssen.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Richten Sie automatisierte Builds ein, die bei jedem Commit oder Pull Request auf den Hauptbranch ausgelöst werden
- Beziehen Sie Kompatibilitäts- und Integrationstests neben Unit-Tests in die CI-Pipeline ein
- Halten Sie die CI-Feedback-Schleife schnell (unter 15 Minuten), um häufige Integration zu fördern
- Setzen Sie trunk-basierte Entwicklung oder kurzlebige Branches durch, um Integrationsdrift zu reduzieren
- Fügen Sie Vertragstests und Schemavalidierung hinzu, um Kompatibilitätsregressionen automatisch zu erkennen
- Überwachen Sie CI-Pipeline-Gesundheitskennzahlen (Erfolgsquote, Dauer, Flakiness) und beheben Sie Verschlechterung zeitnah

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Erkennt Integrations- und Kompatibilitätsprobleme innerhalb von Minuten nach ihrer Einführung
- Reduziert den Schmerz des Mergens langlebiger Branches, indem kleine, häufige Integrationen gefördert werden
- Baut Vertrauen für das Deployment von Legacy-Systemen auf, indem automatisierte Sicherheitsnetze bereitgestellt werden

**Kosten und Risiken:**
- Legacy-Codebasen ohne Tests erfordern erhebliche Vorabinvestition, um CI aussagekräftig zu machen
- Flakige Tests in Legacy-Systemen können das Vertrauen in die CI-Pipeline untergraben
- CI-Infrastruktur erfordert laufende Pflege und Skalierung
- Schnelle Feedback-Schleifen können bei langsamen Legacy-Build- und Testprozessen schwer zu erreichen sein

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Legacy-Java-Monolith hatte einen zweiwöchigen Integrationszyklus, in dem Entwickler Branches mergten und Tage mit der Lösung von Konflikten und Regressionen verbrachten. Das Team führte CI mit automatisierten Builds bei jedem Push ein, beginnend mit einer Smoke-Test-Suite, die acht Minuten lief. Über sechs Monate erweiterten sie die Testabdeckung und verkürzten Feature-Branches auf maximal zwei Tage. Integrationsbezogene Fehler sanken um 65 Prozent, und das Team wechselte von zweiwöchentlichen zu wöchentlichen Releases.
