---
title: Docs as Code
description: Behandlung und Verwaltung von Dokumentation wie Quellcode.
category:
- Communication
- Process
quality_tactics_url: https://qualitytactics.de/en/maintainability/docs-as-code/
problems:
- poor-documentation
- information-decay
- unclear-documentation-ownership
- information-fragmentation
- legacy-system-documentation-archaeology
- incomplete-knowledge
- implicit-knowledge
- tacit-knowledge
- difficult-developer-onboarding
- inconsistent-onboarding-experience
- inadequate-onboarding
- knowledge-gaps
- system-integration-blindness
- communication-risk-within-project
- duplicated-effort
- duplicated-research-effort
- extended-research-time
- knowledge-sharing-breakdown
- language-barriers
- legal-disputes
- mentor-burnout
- new-hire-frustration
- rapid-team-growth
- team-churn-impact
- unproductive-meetings
- communication-breakdown
- duplicated-work
- inconsistent-knowledge-acquisition
- knowledge-dependency
- poor-communication
- staff-availability-issues
- unclear-sharing-expectations
- implementation-partner-dependency
layout: solution
lang: de
en_slug: documentation-as-code
related_solutions:
- slug: living-documentation
  similarity: 0.85
- slug: architecture-documentation
  similarity: 0.85
- slug: architecture-decision-records
  similarity: 0.8
- slug: api-documentation
  similarity: 0.8
- slug: ci-cd-pipeline
  similarity: 0.8
- slug: knowledge-sharing-practices
  similarity: 0.8
---

## Description

Docs as Code speichert Dokumentation als Klartext-Dateien im selben versionskontrollierten Repository wie den Code, den sie beschreiben, überprüft durch denselben Pull-Request-Workflow, statt in einem separaten Wiki, das aus dem Takt gerät, sobald niemand mehr aktiv daran arbeitet. Legacy-Systeme hängen stark von institutionellem Wissen ab, das nur in den Köpfen einiger weniger Menschen und in verstreuten, veralteten Wiki-Seiten und E-Mail-Threads existiert, was genau das Material ist, das diese Praxis erfassen soll — beginnend mit den kritischsten undokumentierten Bereichen und der Anforderung, dass jeder verhaltensändernde Pull Request die relevanten Docs zusammen mit sich aktualisiert. Weil sowohl Code als auch Dokumentation durch dieselbe Commit-Historie und denselben Review-Prozess laufen, bleibt die Aufzeichnung konstruktionsbedingt aktuell statt allein durch Disziplin, und sie hinterlässt eine dauerhafte, autorisierte Spur dessen, was gelernt wurde und wann — wertvoll noch lange, nachdem die Person, die es gelernt hat, weitergezogen ist.

## How to Apply ◆

> In Legacy-Kontexten verwandelt Docs as Code undokumentiertes institutionelles Wissen in versionierte, überprüfbare Artefakte, die neben dem Code leben, den sie beschreiben.

- Beginnen Sie damit, die kritischsten undokumentierten Bereiche zu identifizieren: Integrationspunkte, Deployment-Prozeduren und Komponentengrenzen, die nur ein oder zwei Personen kennen. Schreiben Sie diese zuerst, in Markdown-Dateien, die neben dem relevanten Code liegen.
- Speichern Sie alle Dokumentation im selben Versionskontroll-Repository wie den Anwendungsquellcode. Für Legacy-Systeme mit mehreren Repositories platzieren Sie Dokumentation am nächsten zum Code, den sie beschreibt — ein `docs/`-Ordner eines Services ist besser als ein zentralisiertes Wiki weit vom Code entfernt.
- Etablieren Sie eine Pull-Request-Norm: Jede Änderung, die Verhalten, Konfiguration oder Integrationsverträge ändert, muss eine Dokumentationsaktualisierung enthalten. Dies ist besonders wichtig während Modernisierungsbemühungen, bei denen sich Schnittstellen und Annahmen häufig ändern.
- Führen Sie leichtgewichtige automatisierte Prüfungen in die CI-Pipeline ein: mindestens Erkennung defekter Links und einen Rechtschreibprüfer. Diese fangen die Dokumentationsverfall ab, unter dem Legacy-Systeme typischerweise leiden, ohne erheblichen Vorabaufwand zu erfordern.
- Nutzen Sie Klartextformate (Markdown reicht für die meisten Fälle), sodass die Dokumentation mit denselben Werkzeugen bearbeitbar ist, die Entwickler bereits nutzen. Entfernen Sie die Hürde, zu einem browserbasierten Wiki-Editor zu wechseln.
- Behandeln Sie bestehende verstreute Dokumentation (E-Mail-Threads, Wiki-Seiten, Word-Dokumente, in gemeinsamen Laufwerken vergrabene Diagramme) als Quellmaterial. Migrieren Sie wertvollen Inhalt in versionskontrollierte Dateien und verwerfen oder archivieren Sie den Rest. Tun Sie dies schrittweise, nicht als Big-Bang-Migration.
- Fügen Sie der CI/CD-Pipeline einen Dokumentations-Build-Schritt mittels eines statischen Site-Generators wie MkDocs oder Docusaurus hinzu, sodass der aktuelle Stand der Dokumentation immer ohne manuellen Aufwand veröffentlicht und zugänglich ist.
- Wenn Sie Legacy-Archäologie betreiben — per Reverse Engineering herausfinden, was das System tatsächlich tut —, dokumentieren Sie Befunde sofort im Repository. Jede Entdeckung ist ein Commit. Dies erzeugt eine Prüfspur dessen, was gelernt wurde und wann.

## Tradeoffs ⇄

> Die Gewinne bei Dokumentationsqualität und -aktualität kommen mit echter Vorabinvestition, besonders wenn das Team bereits durch Legacy-System-Anforderungen ausgelastet ist.

**Vorteile:**

- Dokumentation bleibt mit Codeänderungen synchron, weil beide durch denselben Pull-Request-Workflow laufen, was die chronische Veraltung reduziert, unter der Legacy-System-Wikis leiden.
- Jede Dokumentationsänderung hat einen Autor, eine Review-Aufzeichnung und eine Commit-Nachricht, die erklärt, warum die Änderung vorgenommen wurde — kritisches institutionelles Gedächtnis für Systeme, deren ursprüngliche Autoren längst gegangen sind.
- Entwickler tragen bereitwilliger bei, weil sie in Werkzeugen arbeiten, die sie bereits kennen: ihrem Editor, Git und der Kommandozeile — nicht einer separaten Wiki-Anwendung, die sie selten besuchen.
- Automatisierte Link-Prüfung und CI-Validierung fangen defekte Referenzen und fehlende Abschnitte ab, bevor sie die nächste Person in die Irre führen, die die Dokumentation während eines Vorfalls oder Onboardings liest.
- Die vollständige Historie der Dokumentation offenbart, wie sich das System entwickelt hat, einschließlich Entscheidungen, die getroffen und später rückgängig gemacht wurden — unschätzbarer Kontext, um zu verstehen, warum Legacy-Code so aussieht, wie er aussieht.

**Kosten und Risiken:**

- Legacy-Systeme haben oft nicht-technische Stakeholder (Business-Analysten, Compliance-Beauftragte, Betriebspersonal), die Dokumentation in Wikis pflegen. Sie zu bitten, Git und Markdown zu nutzen, schafft eine erhebliche Lernkurve und potenziellen Widerstand.
- Die anfängliche Migration verstreuter Legacy-Dokumentation in versionskontrollierte Dateien ist arbeitsintensiv und konkurriert mit Feature- und Wartungsarbeit. Teams unterschätzen oft, wie viel Material über gemeinsame Laufwerke, E-Mail und informelle Wikis existiert.
- Ohne aktive Durchsetzung im Code-Review verfällt die Pull-Request-mit-Docs-Norm unter Lieferdruck schnell. Legacy-Systeme unter aktivem Feuerlöschen sind besonders anfällig dafür, dass Dokumentation als „wir holen das später nach" übersprungen wird.
- Build-Pipelines und Toolchains für statische Site-Generatoren fügen Komplexität hinzu, die das Team pflegen muss. Für Legacy-Organisationen mit begrenzter DevOps-Reife ist dies eine nicht triviale operative Last.
- Klartextformate fehlt die Diagramm-Einbettung und reiche Formatierung, die manche Legacy-Dokumentation echt braucht. Teams brauchen möglicherweise zusätzliches Tooling (z. B. PlantUML, Mermaid), um diagrammlastige Dokumentation zu ersetzen, die zuvor in Werkzeugen wie Confluence oder Visio gepflegt wurde.

## How It Could Be

> Legacy-Modernisierungsprojekte gelingen oder scheitern am institutionellen Wissenstransfer, und Docs as Code bietet einen dauerhaften Mechanismus, um zu erfassen, was das Team unterwegs lernt.

Ein Finanzdienstleistungsunternehmen unternahm eine mehrjährige Migration eines monolithischen Zahlungsverarbeitungssystems zu einer Reihe kleinerer Services. Ihre bestehende Dokumentation bestand aus einem zuletzt 2019 aktualisierten Confluence-Wiki und einer Sammlung von Word-Dokumenten auf einem gemeinsamen Netzlaufwerk. Das Modernisierungsteam etablierte eine Richtlinie, dass jede analysierte Komponente in Markdown-Dateien dokumentiert wird, die ins Repository committet werden. Über achtzehn Monate bauten sie ein lebendes Architekturdokument, das das tatsächliche Verhalten von Integrationspunkten erfasste, nicht das beabsichtigte Verhalten aus veralteten Spezifikationen. Als ein Schlüsselarchitekt mitten im Projekt ausschied, absorbierte die Dokumentation das meiste von dem, was er wusste, und das Onboarding seines Nachfolgers dauerte Tage statt Monate.

Eine Regierungsbehörde, die ein vierzig Jahre altes COBOL-Batch-Verarbeitungssystem pflegte, hatte keine Dokumentation für seine Geschäftsregeln — die Regeln existierten nur im Code und im Gedächtnis dreier bald in Rente gehender Mitarbeiter. Das Team führte eine Reihe strukturierter Wissenserfassungssitzungen durch und transkribierte, was jeder Experte erklärte, in Markdown-Dateien, die im selben Repository wie der COBOL-Quellcode gespeichert wurden. Sie nutzten dann Review-Sitzungen, um die Dokumentation gegen tatsächliches Code-Verhalten zu kreuzprüfen und erstellten Pull Requests mit Korrekturen. Als die Experten in Rente gingen, enthielt das Repository mehrere hundert Seiten Geschäftsregeldokumentation, die von mehreren Personen überprüft und validiert worden war, mit Git-Historie, die zeigte, welcher Experte welchen Abschnitt beigetragen hatte.

Ein E-Commerce-Unternehmen stellte fest, dass jedes Deployment seines Legacy-Auftragsmanagementsystems eine bestimmte Reihe manueller Schritte erforderte, die sich je nach anvisierter Umgebung subtil unterschieden. Dieses Wissen lebte im Kopf eines leitenden Ingenieurs und in einer veralteten Confluence-Seite, der niemand vertraute. Das Team verlegte das Deployment-Runbook als Markdown-Datei ins Repository und verlangte, dass auf jedes Deployment ein Pull Request folgte, der das Runbook aktualisierte, wenn sich etwas geändert hatte. Innerhalb von sechs Monaten war das Runbook maßgeblich, aktuell und vertrauenswürdig — weil es reale Validierung und Review bei jedem Release durchlaufen hatte.
