---
title: Wissenslücken
description: Mangelndes Verständnis von Systemen, Geschäftsanforderungen oder technischen
  Domänen führt zu verlängerter Recherchezeit und suboptimalen Lösungen.
category:
- Communication
- Process
- Team
related_problems:
- slug: skill-development-gaps
  similarity: 0.8
- slug: knowledge-silos
  similarity: 0.75
- slug: knowledge-dependency
  similarity: 0.7
- slug: incomplete-knowledge
  similarity: 0.7
- slug: feature-gaps
  similarity: 0.7
- slug: information-fragmentation
  similarity: 0.7
solutions:
- architecture-decision-records
- documentation-as-code
- knowledge-sharing-practices
- structured-onboarding-program
- api-documentation
- code-comments
- consistent-terminology
- contextual-help
- frequently-asked-questions-faq
- pattern-language
- raising-user-awareness
- security-community
- security-culture
- security-policies-for-users
- security-tests-by-external-parties
- security-training
- ubiquitous-language
- collaborative-problem-solving
- domain-experts
- domain-quiz
- emergency-drills
- interactive-tutorials
- knowledge-base
- personal-support
- plain-language
- threat-intelligence
- user-communities
- video-tutorials
- code-reading-sessions
- application-portfolio-inventory
- domain-immersion
layout: problem
lang: de
en_slug: knowledge-gaps
---

## Description

Wissenslücken treten auf, wenn Teammitgliedern ausreichendes Verständnis der Systeme, mit denen sie arbeiten, der Geschäftsdomäne, der sie dienen, oder der für ihre Aufgaben erforderlichen technischen Ansätze fehlt. Diese Lücken zwingen Entwickler, erhebliche Zeit mit Recherchieren, Experimentieren und Lernen zu verbringen, statt Lösungen effizient zu implementieren. Wissenslücken können auf mehreren Ebenen existieren, vom Verständnis spezifischer APIs oder Frameworks bis zum Verständnis komplexer Geschäftsregeln oder Systemarchitekturen, und sie verstärken sich über die Zeit, während sich Systeme weiterentwickeln und institutionelles Wissen verloren geht. Dieses Problem führt zu Wissenssilos, Single Points of Failure und verringerter Team-Resilienz. In schweren Fällen kann es zu einem "Bus-Faktor" von eins führen, bei dem der Verlust eines einzelnen Teammitglieds für das Projekt katastrophal wäre.

## Indicators ⟡
- Entwickler stellen häufig grundlegende Fragen zu Systemen, mit denen sie regelmäßig arbeiten
- Implementierungsaufgaben dauern viel länger als erwartet aufgrund von Lernanforderungen
- Lösungen sind suboptimal, weil Entwickler bessere Ansätze nicht kennen
- Teammitglieder vermeiden es, an bestimmten Teilen des Systems zu arbeiten, aufgrund von Wissenslücken
- Neue Features werden implementiert, indem bestehende Muster kopiert werden, ohne zu verstehen warum
- Es gibt keine Dokumentation für das Projekt.
- Die Dokumentation ist veraltet und unzuverlässig.

## Symptoms ▲

- [Verlängerte Recherchezeit](verlaengerte-recherchezeit.md)
<br/>  Entwickler verbringen übermäßig viel Zeit mit der Recherche zu Systemen und Domänen, die sie nicht gut verstehen.
- [Suboptimale Lösungen](suboptimale-loesungen.md)
<br/>  Fehlendes Domänen- oder technisches Wissen führt zu Implementierungsentscheidungen, die nicht der beste Ansatz sind.
- [Langsame Entwicklungsgeschwindigkeit](langsame-entwicklungsgeschwindigkeit.md)
<br/>  Lernanforderungen verlängern die für Implementierungsaufgaben benötigte Zeit erheblich.
- [Wissensabhängigkeit](wissensabhaengigkeit.md)
<br/>  Teammitglieder mit Wissenslücken werden abhängig von den wenigen, die die nötige Expertise besitzen.
- [Erhöhtes Risiko für Fehler](erhoehtes-risiko-fuer-fehler.md)
<br/>  Entwickler, die ohne ausreichendes Systemverständnis arbeiten, führen mehr Defekte ein.

## Causes ▼

- [Zusammenbruch des Wissensaustauschs](zusammenbruch-des-wissensaustauschs.md)
<br/>  Unwirksamer Wissensaustausch lässt Teammitglieder ohne Zugang zu Informationen, die andere besitzen.
- [Informationsverfall](informationsverfall.md)
<br/>  Veraltete oder fehlende Dokumentation zwingt Entwickler, ohne zuverlässiges Referenzmaterial zu arbeiten.
- [Hohe Fluktuation](hohe-fluktuation.md)
<br/>  Häufige Abgänge erfahrener Mitarbeiter führen dazu, dass institutionelles Wissen verloren geht.
- [Wissenssilos](wissenssilos.md)
<br/>  Kritisches Wissen, das bei Einzelpersonen isoliert ist, wird zu einer Lücke für alle anderen im Team.

## Detection Methods ○
- **Lernzeit-Tracking:** Messung der für Recherche vs. Implementierung während Entwicklungsaufgaben aufgewendeten Zeit
- **Fragehäufigkeitsanalyse:** Beobachtung, wie oft Teammitglieder um Hilfe beim Verständnis von Systemkomponenten bitten
- **Implementierungsqualitäts-Reviews:** Identifikation von Lösungen, die mit besserem Domänenwissen verbessert werden könnten
- **Wissens-Audit:** Systematische Bewertung des Teamverständnisses kritischer Systemkomponenten
- **Onboarding-Zeit-Metriken:** Nachverfolgung, wie lange neue Teammitglieder brauchen, um in unterschiedlichen Bereichen produktiv zu werden
- **Bus-Faktor-Analyse:** Identifikation kritischer Komponenten oder Systeme, die nur von ein oder zwei Personen verstanden werden. Bewertung, wie viele kritische Personen, wenn sie entfernt würden, das Projekt schwer beeinträchtigen würden.
- **Code-Review-Beobachtungen:** Beachtung, ob Reviewer häufig grundlegende Konzepte oder Muster erklären, die allgemeines Wissen sein sollten.
- **Post-Mortems/Retrospektiven:** Analyse, ob wiederkehrende Probleme durch besseren Wissensaustausch hätten verhindert werden können.
- **Entwicklerbefragungen:** Befragung von Teammitgliedern zu ihrem Zugang zu notwendigen Informationen und Lernmöglichkeiten sowie ihren Herausforderungen beim Finden von Informationen.
- **Kommunikationsmuster-Analyse:** Beachtung, ob Fragen immer an dieselben wenigen Personen gerichtet werden, oder ob Informationen nur in privaten Kanälen geteilt werden.

## Examples

Ein Team für die Entwicklung von Gesundheitssoftware muss neue Patientendatenschutz-Features implementieren, aber keiner der aktuellen Entwickler hat Erfahrung mit HIPAA-Compliance-Anforderungen. Sie verbringen Wochen mit der Recherche von Vorschriften, der Konsultation mit Rechtsteams und dem Experimentieren mit unterschiedlichen Implementierungsansätzen, bevor sie entdecken, dass ihre gewählte Lösung die Sicherheitsanforderungen tatsächlich nicht erfüllt. Dies führt zu einem kompletten Redesign, das mit ordentlichem Domänenwissen hätte vermieden werden können. Ein weiteres Beispiel betrifft ein Team, das ein Finanzhandelssystem wartet, bei dem die ursprünglichen Entwickler das Unternehmen verlassen haben. Aktuelle Teammitglieder verstehen die grundlegende Codestruktur, aber es fehlt ihnen Wissen über die komplexen Handelsalgorithmen und marktspezifischen Geschäftsregeln. Wenn sie gebeten werden, die Positionsberechnungslogik zu modifizieren, verbringen sie Tage mit dem Lesen undokumentierten Codes und der Recherche finanzieller Konzepte, bevor sie erkennen, dass sie Geschäfts-Stakeholder einbeziehen müssen, um das beabsichtigte Verhalten zu verstehen, was eine eigentlich unkomplizierte Änderung erheblich verzögert.

Ein kritisches Legacy-System wird von einem einzigen Senior-Ingenieur gewartet. Wenn dieser Ingenieur in den Urlaub geht, entsteht ein größerer Fehler, und niemand sonst im Team hat genug Wissen, um ihn schnell zu diagnostizieren und zu beheben, was zu längerer Ausfallzeit führt. In einem anderen Fall entwickeln zwei unterschiedliche Teams innerhalb derselben Organisation unabhängig voneinander ähnliche Microservices, wobei jedes gängige Probleme wie Authentifizierung und Logging von Grund auf löst, ohne sich der Arbeit des anderen oder bestehender interner Bibliotheken bewusst zu sein.

Ein neuer Entwickler tritt dem Team bei und verbringt seinen ersten Monat damit, grundlegende Fragen zum Projekt-Setup und Deployment-Prozess zu stellen – Informationen, die nirgendwo dokumentiert sind und wiederholt von unterschiedlichen Teammitgliedern erklärt werden müssen. Dieses Problem ist besonders akut in Legacy-System-Modernisierungsprojekten, in denen ein Großteil des Wissens des ursprünglichen Systems nur in den Köpfen langjähriger Mitarbeiter existiert. Ohne aktiven Wissenstransfer läuft diese kritische Information Gefahr, verloren zu gehen. Das Problem ist besonders verbreitet in wachsenden Organisationen oder solchen, die einen erheblichen technologischen Wandel durchlaufen, und es beeinträchtigt direkt Skalierbarkeit, Resilienz und das gesamte intellektuelle Kapital des Engineering-Teams.
