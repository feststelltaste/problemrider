---
title: Langsame Entwicklungsgeschwindigkeit
description: Das Team schafft es konsequent nicht, Features und Bugfixes in einem
  vorhersehbaren und akzeptablen Tempo zu liefern, wobei die Gesamtproduktivität systematisch sinkt.
category:
- Business
- Code
- Process
- Team
related_problems:
- slug: reduced-team-productivity
  similarity: 0.8
- slug: slow-feature-development
  similarity: 0.8
- slug: inefficient-development-environment
  similarity: 0.75
- slug: reduced-individual-productivity
  similarity: 0.75
- slug: difficult-developer-onboarding
  similarity: 0.75
- slug: increased-stress-and-burnout
  similarity: 0.7
solutions:
- architecture-roadmap
- development-environment-optimization
- development-workflow-automation
- regression-testing
- microservices-architecture
- trunk-based-development
- delivery-performance-metrics
- fast-feedback-loops
- baseline-measurement
- value-hierarchy
- cost-of-delay
layout: problem
lang: de
en_slug: slow-development-velocity
---

## Description
Langsame Entwicklungsgeschwindigkeit repräsentiert eine anhaltende Verringerung der Fähigkeit des Teams, Features effektiv zu liefern, Bugs zu beheben oder Systeme zu warten. Dieses Problem umfasst sowohl verringerte Produktivität, bei der der Gesamtoutput des Teams systematisch sinkt, als auch das konsequente Versäumnis des Teams, Termine einzuhalten und Wert in vorhersehbarem Tempo zu liefern. Es zeichnet sich durch einen wachsenden Backlog, verpasste Termine, verlängerte Feature-Lieferzeiten und ein allgemeines Gefühl von Frustration und Stagnation im Team aus. Anders als vorübergehende Produktivitätsrückgänge repräsentiert dies einen langfristigen Abstieg, der oft graduell entsteht, während sich technische Schulden anhäufen, die Team-Moral erodiert und Systeme zunehmend schwieriger zu bearbeiten werden, was eine Abwärtsspirale schafft, die die gesamten Geschäftsergebnisse beeinflusst.

## Indicators ⟡
- Das Team verpasst konsequent Sprint-Ziele oder Release-Termine.
- Die Sprint-Velocity nimmt über mehrere Iterationen konsequent ab.
- Der Arbeitsrückstand wächst schneller, als er abgearbeitet wird.
- Es dauert lange, neue Features von der Idee bis zur Produktion zu bringen.
- Features, die früher Tage brauchten, brauchen jetzt Wochen zur Implementierung.
- Es gibt viel Kontextwechsel und Multitasking.
- Entwickler verbringen mehr Zeit mit Debugging und Fehlerbehebung als mit dem Bau neuer Funktionalität.
- Team-Schätzungen für ähnliche Arbeitselemente steigen über die Zeit kontinuierlich.
- Mehr Zeit wird in Meetings verbracht, in denen Probleme diskutiert werden, als sie zu lösen.

## Symptoms ▲

- [Verpasste Termine](verpasste-termine.md)
<br/>  Sinkende Velocity verursacht direkt, dass das Team konsequent geplante Liefertermine verpasst.
- [Verzögerte Wertlieferung](verzoegerte-wertlieferung.md)
<br/>  Wenn die Entwicklungsgeschwindigkeit sinkt, erreicht Geschäftswert Nutzer weit später als erwartet.
- [Entwicklerfrustration und Burnout](entwicklerfrustration-und-burnout.md)
<br/>  Anhaltend langsame Lieferung demoralisiert das Team, während es sich abmüht, bedeutsamen Fortschritt zu machen.

## Causes ▼

- [Kurzfristiger Fokus](kurzfristiger-fokus.md)
<br/>  Durch das konsequente Aufschieben von Investitionen in Code-Gesundheit, Refactoring und Entwicklungswerkzeuge lässt kurzfristiger Fokus technische Schulden und Ineffizienzen sich anhäufen, was die Lieferung über die Zeit systematisch verlangsamt.
- [Hohe technische Schulden](hohe-technische-schulden.md)
<br/>  Angehäufte technische Schulden machen jede Änderung schwieriger und langsamer, was die Geschwindigkeit systematisch verringert.
- [Spaghetticode](spaghetticode.md)
<br/>  Verworrener, unstrukturierter Code macht selbst einfache Änderungen zeitaufwendig, weil Entwickler komplexe Logik verstehen müssen.
- [Schlechte Dokumentation](schlechte-dokumentation.md)
<br/>  Fehlende Dokumentation zwingt Entwickler, exzessive Zeit damit zu verbringen, das System zu verstehen, bevor sie Änderungen vornehmen.
- [Ineffiziente Entwicklungsumgebung](ineffiziente-entwicklungsumgebung.md)
<br/>  Langsame Build-Zeiten, schlechte Werkzeuge und umständliche Entwicklungsworkflows verschwenden Entwicklerzeit für unproduktive Aktivitäten.
- [Overhead durch Kontextwechsel](overhead-durch-kontextwechsel.md)
<br/>  Häufiger Kontextwechsel zwischen Aufgaben fragmentiert die Aufmerksamkeit der Entwickler und verringert den effektiven Output.
- [Review-Engpässe](review-engpaesse.md)
<br/>  Review-Engpässe verlangsamen direkt die Entwicklungsgeschwindigkeit, indem sie verhindern, dass Code gemerged und deployt wird.
- [Analyse-Lähmung](analyse-laehmung.md)
<br/>  Zeit, die in Analyse und Recherche feststeckt, produziert keinen funktionierenden Code, was die effektive Entwicklungsgeschwindigkeit für die Dauer dieser Phase gegen null treibt.
- [Angehäufte Entscheidungsschulden](angehaeufte-entscheidungsschulden.md)
<br/>  Ein wachsender Rückstand ungelöster, voneinander abhängiger Entscheidungen zwingt jede nachfolgende Änderung, zusätzliche Beschränkungen zu navigieren, was die Lieferung verlangsamt.

## Detection Methods ○
- **Velocity-Verfolgung:** Verfolgung der Team-Velocity über die Zeit, um zu sehen, ob sie sich verbessert oder verschlechtert. Überwachung von Sprint-Velocity, abgeschlossenen Story Points oder gelieferten Features über die Zeit.
- **Zykluszeit-Analyse:** Analyse der Zeit, die eine Aufgabe von Anfang bis Ende braucht. Messung der Zeit von der Feature-Anfrage bis zum Deployment für ähnliche Arten von Arbeit.
- **Zeitanalyse:** Verfolgung, wie Entwickler ihre Zeit verbringen (Coding vs. Debugging vs. Meetings vs. Recherche).
- **Entwicklerbefragungen:** Regelmäßiges Feedback zu Hindernissen, Frustrationen und Produktivitätsbarrieren.
- **Arbeitselement-Analyse:** Vergleich aktueller Schätzungen und tatsächlicher Fertigstellungszeiten mit historischen Daten.

## Examples
Ein Team arbeitet an einem neuen Feature für sein Produkt. Es schätzt, dass es zwei Sprints zur Fertigstellung braucht. Nach vier Sprints ist das Feature jedoch immer noch nicht fertig. Das Team wird konstant durch mangelnde klare Anforderungen, eine komplexe Codebasis und eine langsame Entwicklungsumgebung blockiert. Infolgedessen kann es keinen Fortschritt machen, und das Feature wird schließlich abgesagt.

Ein Entwicklungsteam, das eine Legacy-E-Commerce-Plattform wartet, erlebt über 18 Monate graduell sinkende Velocity. Anfangs dauerte das Hinzufügen neuer Zahlungsmethoden 2 Wochen, aber jetzt brauchen ähnliche Features 6 Wochen aufgrund der Komplexität der Integration mit einem zunehmend verworrenen Zahlungsverarbeitungssystem. Entwickler verbringen 60 % ihrer Zeit mit der Fehlerbehebung von Integrationsproblemen, dem Lesen undokumentierten Codes und dem Umgehen von Beschränkungen der bestehenden Architektur. Was früher ein produktives Team war, das 2-3 Hauptfeatures pro Monat lieferte, kämpft jetzt darum, ein Feature in derselben Zeitspanne fertigzustellen. Ein weiteres Beispiel betrifft ein Team, das ein Kundensupportsystem wartet, dessen Codebasis sich so viele technische Schulden angehäuft hat, dass jede Änderung das Berühren mehrerer nicht verwandter Module erfordert. Ein einfaches Feature wie das Hinzufügen eines neuen Felds zu einem Support-Ticket-Formular erfordert jetzt Änderungen an 12 verschiedenen Dateien, umfangreiches Testen, um das Brechen bestehender Funktionalität zu vermeiden, und sorgfältige Koordination, um Konflikte mit anderer laufender Arbeit zu vermeiden.
