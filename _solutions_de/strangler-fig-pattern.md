---
title: Strangler-Fig-Pattern
description: Schrittweiser Ersatz von Legacy-Systemen durch Weiterleitung
  von Traffic zu neuen Implementierungen.
category:
- Architecture
quality_tactics_url: https://qualitytactics.de/en/maintainability/strangler-fig-pattern/
problems:
- monolithic-architecture-constraints
- legacy-business-logic-extraction-difficulty
- strangler-fig-pattern-failures
- stagnant-architecture
- system-stagnation
- technology-lock-in
- fear-of-breaking-changes
- fear-of-change
- architectural-mismatch
- inability-to-innovate
- technical-architecture-limitations
- high-maintenance-costs
- obsolete-technologies
layout: solution
lang: de
en_slug: strangler-fig-pattern
related_solutions:
- slug: feature-flags
  similarity: 0.8
- slug: event-driven-architecture
  similarity: 0.8
- slug: modularization-and-bounded-contexts
  similarity: 0.75
- slug: microservices
  similarity: 0.75
- slug: ci-cd-pipeline
  similarity: 0.75
- slug: architecture-roadmap
  similarity: 0.75
---

## Description

Das Strangler-Fig-Pattern ersetzt ein Legacy-System eine Fähigkeit nach der anderen, indem Traffic durch einen Proxy geleitet wird, der jedes migrierte Stück Funktionalität zu seiner neuen Implementierung sendet, während alles noch nicht Migrierte unverändert weiter zum Legacy-System fließt. Dies vermeidet direkt die Alles-oder-nichts-Neuschreibung, die in Legacy-Kontexten unverhältnismäßig oft scheitert, genau weil die versteckte Komplexität, die eine Big-Bang-Neuschreibung vollständig lösen muss, bevor sie live gehen kann, genau das ist, was niemand im Voraus akkurat abschätzen kann. Der Beginn mit Fähigkeiten, die sowohl häufig geändert als auch sauber vom Rest des Systems trennbar sind, gibt dem Team genau dort Entlastung, wo der Legacy-Code derzeit am meisten schmerzt, während gleichzeitig die Routing- und Charakterisierungstest-Muskeln aufgebaut werden, die benötigt werden, bevor der zutiefst verworrene Kern angegangen wird — obwohl das Pattern ohne ein festes Mandat, jedes migrierte Stück außer Betrieb zu nehmen, genauso leicht zwei dauerhafte, nebeneinander laufende Systeme statt des beabsichtigten Übergangs produziert.

## How to Apply ◆

> In der Legacy-Systemmodernisierung ersetzt das Strangler-Fig-Pattern die gefährliche Alles-oder-nichts-Neuschreibung durch eine kontrollierte, Fähigkeit-für-Fähigkeit-Migration, die das Legacy-System durchgehend voll funktionsfähig hält.

- Platzieren Sie eine Routing-Schicht — einen Reverse-Proxy, ein API-Gateway oder einen Message-Router — vor dem Legacy-System, bevor Sie eine einzige Zeile Ersatzcode schreiben. Diese Schicht ist die grundlegende Investition, auf die sich alle nachfolgenden Migrationen verlassen werden; bauen Sie sie so, dass sie schrittweise Traffic-Verlagerung und schnellen Rollback unterstützt.
- Kartieren Sie die Fähigkeiten des Legacy-Systems und identifizieren Sie natürliche Migrationsgrenzen mittels Abhängigkeitsanalyse oder Event Storming. Suchen Sie nach Clustern von Funktionalität mit begrenzter Datenkopplung zum Rest des Systems — dies sind die sichersten Ausgangspunkte.
- Priorisieren Sie sich häufig ändernde Fähigkeiten für frühe Migration. Eine Geschäftsregel, die monatliche Updates an einem schlecht strukturierten Legacy-Modul erfordert, ist ein Hauptkandidat: Das Team erhält sofortige Entlastung von der Legacy-Codebasis genau dort, wo es am meisten wehtut.
- Schreiben Sie, bevor Sie irgendeinen Ersatz bauen, Charakterisierungstests gegen das tatsächliche Verhalten des Legacy-Systems — einschließlich seiner undokumentierten Eigenheiten und Grenzfälle. Diese Tests werden zu den Abnahmekriterien, die die neue Implementierung erfüllen muss, und zum Sicherheitsnetz während der Umschaltung.
- Nutzen Sie Shadow-Traffic oder Canary-Deployments für jede Fähigkeitsumschaltung: Senden Sie zunächst einen kleinen Prozentsatz echter Anfragen an sowohl die alte als auch die neue Implementierung, vergleichen Sie die Ausgaben, und verlagern Sie Traffic erst vollständig, nachdem Diskrepanzen gelöst sind.
- Etablieren Sie eine strikte Disziplin, Legacy-Dead-Code unmittelbar nach jeder erfolgreichen Fähigkeitsmigration zu entfernen. Ohne dies besteht das Legacy-System unbegrenzt neben dem neuen fort, was die Wartungslast verdoppelt statt sie zu reduzieren.
- Planen Sie Datenmigration als erstklassigen Belang. Identifizieren Sie frühzeitig, ob Fähigkeiten die Legacy-Datenbank vorübergehend teilen können, Datensynchronisation erfordern oder ihre Daten ab dem Moment der Umschaltung unabhängig besitzen können.
- Setzen Sie einen festen Zeitplan zum Abschluss jeder Migrationsphase und kommunizieren Sie ihn über die beteiligten Teams hinweg. Ohne organisatorisches Engagement zur Außerbetriebnahme wird das Pattern zu einer dauerhaften Dual-System-Architektur statt einer Übergangsarchitektur.

## Tradeoffs ⇄

> Das Strangler-Fig-Pattern tauscht die Geschwindigkeit und Einfachheit einer Clean-Room-Neuschreibung gegen Resilienz und kontinuierliche Wertlieferung während einer Migration, die sich über Jahre erstrecken kann.

**Vorteile:**

- Beseitigt das existenzielle Risiko von Big-Bang-Neuschreibungen, die in Legacy-Modernisierungskontexten, wo versteckte Komplexität die Norm ist, häufiger scheitern als gelingen.
- Liefert sichtbaren, messbaren Modernisierungsfortschritt an Stakeholder mit jeder migrierten Fähigkeit und erhält organisatorischen Schwung über eine mehrjährige Migration hinweg.
- Bewahrt die Fähigkeit, jede einzelne Fähigkeit zum Legacy-System zurückzurollen, wenn die neue Implementierung unerwartete Probleme offenbart, und begrenzt den Blast-Radius jedes Migrationsschritts.
- Erlaubt dem Team, aus frühen Migrationen zu lernen — Routing-Muster, Datenmigrationstechniken und Testansätze etablierend —, bevor die komplexesten und verworrensten Teile des Legacy-Systems angegangen werden.
- Reduziert den Druck, das Legacy-System vollständig zu verstehen, bevor begonnen wird: Das Team lernt es Fähigkeit für Fähigkeit, baut Wissen inkrementell neben der neuen Implementierung auf.

**Kosten und Risiken:**

- Der gleichzeitige Betrieb zweier Systeme erhöht Infrastrukturkosten, betriebliche Komplexität und die kognitive Last für das für beide verantwortliche Team während der Übergangsperiode.
- Die Routing-Schicht führt eine neue architektonische Komponente ein, die überwacht, gepflegt und hochverfügbar gehalten werden muss, da der gesamte Traffic durch sie fließt.
- Datensynchronisation zwischen Legacy- und neuen Datenspeichern ist technisch schwierig und eine häufige Quelle von Konsistenzfehlern, besonders für Fähigkeiten mit hohem Schreibvolumen oder komplexer Transaktionssemantik.
- Die Migration kann nach frühen schnellen Erfolgen ins Stocken geraten: Die ersten zu migrierenden Fähigkeiten sind typischerweise die saubersten, während zutiefst verworrene Module im Kern des Legacy-Systems sich als extrem schwierig sauber zu extrahieren erweisen könnten.
- Ohne ein starkes Außerbetriebnahme-Mandat könnten Teams die Dringlichkeit verlieren, die Migration abzuschließen, sobald der unmittelbare Schmerz gelindert ist, was beide Systeme unbegrenzt laufen lässt.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie sich das Strangler-Fig-Pattern in echten Legacy-Modernisierungssituationen entfaltet, von einfachen Routing-Änderungen bis zu komplexen Datenmigrationen.

Ein regionaler Versicherer, der ein in den 1990er-Jahren gebautes monolithisches Policenverwaltungssystem betrieb, musste sein Kunden-Self-Service-Portal modernisieren, ohne den täglichen Betrieb zu stören. Das Team platzierte ein API-Gateway vor dem Legacy-System und begann mit der Migration der einfachsten Fähigkeit zuerst: Policendokumentenabruf. Die neue Implementierung zog Dokumente aus einem modernen Dokumentenspeicher, während das Gateway weiterhin alle anderen Anfragen — Prämienberechnungen, Nachträge, Schäden — an das Legacy-Backend routete. Über achtzehn Monate migrierte das Team eine Fähigkeit nach der anderen, und die Migration des Policendokumentenabrufs gab ihnen die Vorlage, die sie auf jede nachfolgende anwandten. Das Legacy-System handhabte schließlich nur noch die am tiefsten gekoppelten Buchhaltungsfunktionen, bevor die letzte Migration abgeschlossen wurde.

Ein europäisches Logistikunternehmen, das ein Frachtdisposition-System betrieb, das Routenplanung, Fahrerplanung und Rechnungsgenerierung in einem einzigen Monolithen mischte, sah wachsende Nachfrage von mobilen Anwendungen nach einer modernen REST-API. Das Legacy-System legte nur ein proprietäres Thick-Client-Protokoll offen. Das Team baute eine dünne Adapterschicht, die REST-Aufrufe in das Legacy-Protokoll übersetzte, was mobilen Clients erlaubte, sich zu verbinden, während die eigentliche Migration voranschritt. Über die folgenden zwei Jahre wurde die Routenplanungs-Engine — die wertvollste und am häufigsten modifizierte Komponente — in einen unabhängigen Dienst extrahiert. Die Adapterschicht wurde neu konfiguriert, um Planungsanfragen direkt an den neuen Dienst zu routen, während alle anderen Anfragen weiterhin durch den Adapter in das Legacy-System liefen, was dem Unternehmen moderne Planungsfähigkeiten gab, ohne den Dispositionsbetrieb zu unterbrechen.

Ein öffentliches Versorgungsunternehmen, das ein jahrzehntealtes Kundenabrechnungssystem verwaltete, versuchte in zehn Jahren zweimal eine vollständige Neuschreibung, beide Male wurde die Anstrengung nach zwei Jahren Investition ohne Produktionsfreigabe aufgegeben. Beim dritten Versuch mandatierte die Führung den Strangler-Fig-Ansatz. Das Team begann mit der Zählerablese-Erfassung, die klar definierte Eingaben und keinen gemeinsam genutzten Zustand mit dem Rest der Abrechnungslogik hatte. Nach einer erfolgreichen Migration dieser Fähigkeit wechselten sie zur Nutzungsberechnung nur für Geschäftskonten — eine kleinere Teilmenge des Gesamtproblems. Jede Migration baute Vertrauen auf und verfeinerte den Datensynchronisationsansatz des Teams. Drei Jahre später wurde der Legacy-Abrechnungskern schließlich außer Betrieb genommen, und zum ersten Mal in der Geschichte der Organisation produzierte der Übergang keine Abrechnungsstörung für Kunden.
