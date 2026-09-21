---
title: Standardsoftware
description: Nutzung erprobter Standardsoftware, statt gewöhnliche
  Funktionalität selbst zu entwickeln.
category:
- Architecture
- Dependencies
problems:
- maintenance-overhead
- high-maintenance-costs
- maintenance-cost-increase
- obsolete-technologies
- legacy-skill-shortage
- increased-cost-of-development
- slow-feature-development
- technology-lock-in
- excessive-customization
- reimplemented-standard-functionality
layout: solution
lang: de
en_slug: standard-software
related_solutions:
- slug: consistent-user-interface
  similarity: 0.7
- slug: consistent-terminology
  similarity: 0.7
- slug: customizing
  similarity: 0.7
- slug: strangler-fig-pattern
  similarity: 0.7
- slug: emulation
  similarity: 0.65
- slug: boring-technologies
  similarity: 0.65
---

## Description

Standardsoftware bedeutet, eine selbst gebaute, hausinterne Implementierung gängiger Funktionalität durch ein ausgereiftes, weit verbreitetes kommerzielles oder Open-Source-Produkt zu ersetzen, das dasselbe Problem bereits für viele andere Organisationen löst. Legacy-Systeme sammeln maßgeschneiderte Lösungen für Probleme an — Authentifizierung, Berichterstattung, Workflow-Orchestrierung, Terminplanung —, die selbst zum Zeitpunkt ihres Baus keine wettbewerbsdifferenzierenden Faktoren waren, aber genau deshalb dazu wurden, weil jemand sich entschied, neu zu erfinden statt zu übernehmen, und die resultierende maßgeschneiderte Komponente erfordert nun spezialisiertes, knappes internes Wissen, um am Laufen zu bleiben. Diese Lösung zielt auf genau dieses Muster: Sie identifiziert Funktionalität, die generisch statt geschäftsdifferenzierend ist, und ersetzt die laufende Last der Pflege einer maßgeschneiderten Implementierung durch die kontinuierliche Investition eines Anbieters oder einer Community in Sicherheitspatches, Feature-Entwicklung und breiter verfügbares Fachwissen. In der Modernisierungsarbeit ist dies oft einer der wirkungsvollsten verfügbaren Schritte, weil er Single Points of Failure beseitigt, die an den einen Entwickler gebunden sind, der ein obskures maßgeschneidertes Modul versteht, während er Entwicklungskapazität freisetzt, um sich auf die Teile des Systems zu konzentrieren, die das Geschäft tatsächlich differenzieren. Der Kompromiss ist eine Verschiebung von Build-Zeit-Flexibilität zu Anbieterabhängigkeit: Standardsoftware könnte nicht jeden Legacy-Grenzfall replizieren, was eine ehrliche Bewertung erzwingt, ob ein Grenzfall eine echte Anforderung oder einfach ein Artefakt dessen ist, wie das Legacy-System zufällig gebaut wurde, und die Migration selbst erfordert sorgfältige Daten- und Integrationsarbeit, die leicht unterschätzt wird.

## How to Apply ◆

> Legacy-Systeme enthalten häufig maßgeschneiderte Implementierungen von Funktionalität, die jetzt als ausgereifte, gut gepflegte Standardsoftware verfügbar ist — das Ersetzen dieser maßgeschneiderten Komponenten reduziert den Wartungsaufwand erheblich.

- Prüfen Sie das Legacy-System, um maßgeschneiderte Komponenten zu identifizieren, die Funktionalität replizieren, die in erprobter Standardsoftware verfügbar ist, wie maßgeschneiderte Authentifizierungssysteme, Berichtsgeneratoren, Workflow-Engines oder Planungs-Frameworks.
- Bewerten Sie Standardsoftware-Kandidaten gegen die spezifischen Anforderungen, die in der Legacy-Implementierung kodiert sind, mit besonderer Aufmerksamkeit auf Grenzfälle und Anpassungen, die möglicherweise nicht von vornherein unterstützt werden.
- Planen Sie Migrationen von maßgeschneiderter zu Standardsoftware inkrementell, wobei die maßgeschneiderte und die Standardlösung während Übergangsperioden parallel laufen, um Verhaltensparität zu validieren.
- Priorisieren Sie den Ersatz maßgeschneiderter Komponenten, die am teuersten in der Wartung sind, das meiste spezialisierte Wissen erfordern oder das größte Sicherheitsrisiko darstellen.
- Akzeptieren Sie, dass Standardsoftware möglicherweise nicht 100 % des Legacy-Verhaltens repliziert — bewerten Sie, ob die nicht unterstützten Grenzfälle noch echte Anforderungen oder Legacy-Artefakte sind, die eliminiert werden können.
- Verhandeln Sie Support- und Wartungsvereinbarungen für Standardsoftware, um sicherzustellen, dass die Organisation Zugang zu Anbieter-Fachwissen und rechtzeitigen Sicherheitspatches hat.

## Tradeoffs ⇄

> Standardsoftware beseitigt Wartungslast für gängige Funktionalität, führt aber Anbieterabhängigkeiten ein und könnte die Anpassung von Geschäftsprozessen erfordern.

**Vorteile:**

- Reduziert den Wartungsaufwand für Funktionalität, die kein wettbewerbsdifferenzierender Faktor ist, dramatisch und setzt Entwickler frei, sich auf geschäftskritische maßgeschneiderte Features zu konzentrieren.
- Profitiert von der laufenden Investition des Anbieters in Sicherheitspatches, Performance-Verbesserungen und Feature-Entwicklung.
- Reduziert das Risiko von Wissensverlust, da Standardsoftware breitere Community-Dokumentation und verfügbares Fachwissen hat im Vergleich zu maßgeschneiderten Legacy-Komponenten.
- Beschleunigt die Einarbeitung von Entwicklern, weil Teammitglieder eher Erfahrung mit Standardwerkzeugen haben.

**Kosten und Risiken:**

- Führt Anbieterabhängigkeit ein und das Risiko, dass der Anbieter das Produkt einstellt, Lizenzbedingungen ändert oder Preise erhöht.
- Standardsoftware könnte Workflow-Beschränkungen auferlegen, die erfordern, dass die Organisation ihre Prozesse anpasst, was auf Widerstand bei Nutzern stoßen kann, die an Legacy-Verhalten gewöhnt sind.
- Die Migration von maßgeschneiderter zu Standardsoftware erfordert sorgfältige Datenmigration und Integrationsarbeit, die oft unterschätzt wird.
- Übermäßiges Verlassen auf Standardsoftware für Kerngeschäftslogik kann wettbewerbliche Differenzierung und Flexibilität einschränken.

## How It Could Be

> Das folgende Szenario veranschaulicht die Vorteile und Herausforderungen des Ersatzes maßgeschneiderter Legacy-Komponenten durch Standardsoftware.

Ein mittelständisches Fertigungsunternehmen hatte 12 Jahre lang ein maßgeschneidertes ERP-Modul für die Bestandsverwaltung gepflegt. Das Modul erforderte zwei Vollzeitentwickler zur Wartung, lief auf einem veralteten Anwendungsserver und konnte nur von einem einzigen Entwickler modifiziert werden, der seine komplexe Stored-Procedure-Architektur verstand. Nach der Bewertung von drei kommerziellen Bestandsverwaltungssystemen wählte das Unternehmen eines aus, das 85 % ihrer Anforderungen von vornherein abdeckte. Die verbleibenden 15 % bestanden aus maßgeschneiderten Etikettierungs-Workflows, die das Team als Erweiterungen implementierte. Die Migration dauerte acht Monate, beseitigte aber die laufenden Wartungskosten des maßgeschneiderten Moduls und entfernte einen kritischen Single Point of Failure im Entwicklungsteam. Zwei Jahre später hatte der Standardsoftware-Anbieter Berichtsfunktionen geliefert, für deren Bau das maßgeschneiderte System nie die Ressourcen gehabt hätte.
