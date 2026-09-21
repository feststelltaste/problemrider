---
title: Management technischer Schulden
description: Systematische Identifikation, Nachverfolgung und
  Priorisierung technischer Schulden.
category:
- Process
- Management
- Architecture
quality_tactics_url: https://qualitytactics.de/en/maintainability/technical-debt-management/
problems:
- high-technical-debt
- invisible-nature-of-technical-debt
- difficulty-quantifying-benefits
- modernization-roi-justification-failure
- short-term-focus
- refactoring-avoidance
- workaround-culture
- accumulation-of-workarounds
- increasing-brittleness
- brittle-codebase
- competing-priorities
- constant-firefighting
- maintenance-overhead
- maintenance-cost-increase
- high-maintenance-costs
- accumulated-decision-debt
- feature-creep-without-refactoring
- increased-technical-shortcuts
- reduced-predictability
- system-stagnation
layout: solution
lang: de
en_slug: technical-debt-backlog
related_solutions:
- slug: debt-classification
  similarity: 0.85
- slug: functional-debt-management
  similarity: 0.8
- slug: debt-remediation-estimation
  similarity: 0.8
- slug: technical-debt-assessment
  similarity: 0.8
- slug: code-metrics
  similarity: 0.8
- slug: debt-accrual-analysis
  similarity: 0.8
---

## Description

Ein Backlog technischer Schulden erfasst jedes bekannte Schuldenstück als konkretes, geschäftsauswirkungsgeschätztes Element — nicht "der Nutzerservice ist unordentlich", sondern eine spezifische Beschreibung dessen, was verworren ist und was es kostet — und priorisiert es genauso, wie jede andere Arbeit priorisiert wird, statt Schulden als unsichtbare Bremse zu belassen, die jeder spürt, auf die aber niemand zeigen kann. Legacy-Systeme sammeln genau diese Art von Schulden in einem Ausmaß an, das ein anfängliches vollständiges Inventar genuin überwältigend macht, weshalb das Backlog in eine handhabbare, wirkungsstärkste-zuerst-Sequenz triagiert werden muss, statt als eine einzige Masse behandelt zu werden, die geräumt werden muss, bevor irgendetwas anderes geschehen kann. Schulden auf diese Weise sichtbar zu machen ist der gesamte Wert des Backlogs und auch sein Hauptrisiko in politischer Hinsicht, da Stakeholder, die die angesammelten Kosten nie zuvor sehen mussten, es möglicherweise nicht begrüßen, endlich die Rechnung gezeigt zu bekommen — aber ein fester Prozentsatz jedes Sprints, der für Schuldenreduzierung verpflichtet ist, gestützt durch konkrete Geschäftsauswirkungszahlen, ist es, was diese Sichtbarkeit in anhaltende Investition verwandelt statt in einen einmaligen Bericht, der nichts ändert.

## How to Apply ◆

> In Legacy-Systemen, wo technische Schulden enorm, unsichtbar und über Jahre angesammelt sind, ist die Schaffung eines verwalteten Backlogs die Voraussetzung für jede systematische Verbesserung — ohne es ist Sanierung reaktiv und endlos.

- Führen Sie eine anfängliche Schuldenentdeckungsanstrengung durch, die statische Analyse-Scans, Architektur-Reviews und strukturierte Interviews mit den Entwicklern kombiniert, die wissen, welche Module sie fürchten anzufassen; behandeln Sie die Ausgabe als Inventar, nicht als sofortige Arbeitsliste.
- Erfassen Sie jedes Schuldenelement mit einer konkreten Beschreibung (nicht "der Nutzerservice ist unordentlich", sondern "der Nutzerservice enthält Authentifizierung, Profilverwaltung und Audit-Logging in einer einzigen Klasse von 800 Zeilen"), einer geschätzten Geschäftsauswirkung und einem groben Sanierungsaufwand — dies ist die minimale Information, die für Priorisierung benötigt wird.
- Zeichnen Sie Schuldenelemente auf einer Zwei-mal-Zwei-Matrix von Auswirkung versus Aufwand auf; adressieren Sie hochwirkungsvolle, niedrigaufwändige Elemente zuerst als schnelle Erfolge, die Schwung aufbauen und Wert gegenüber Stakeholdern demonstrieren.
- Priorisieren Sie Schulden in Modulen, die aktiv geändert werden; Schulden in stabilem, unangetastetem Legacy-Code kosten nichts, sie zu belassen, während Schulden in häufig modifiziertem Code sich zusammensetzende Zinsen mit jedem Sprint zahlen.
- Integrieren Sie Schuldenreduzierung in die reguläre Entwicklungstaktung, indem Sie einen festen Prozentsatz der Sprint-Kapazität (üblicherweise 20 %) für Schuldenelemente zuweisen, was es zu einer ständigen Verpflichtung macht, statt zu einer konkurrierenden Priorität.
- Wenden Sie die Pfadfinderregel an — beheben Sie kleine Schuldenelemente, wann immer ein Entwickler betroffenen Code anfasst — als grundlegende Praxis, die Schulden inkrementell reduziert, ohne dedizierte Sprints zu erfordern.
- Präsentieren Sie das Schulden-Backlog dem Management in Geschäftsbegriffen: "Dieses Modul fügt jedem Feature, das wir im Checkout-Flow bauen, zwei Tage Overhead hinzu" oder "Diese Schulden verursachten letztes Quartal drei Produktionsvorfälle mit Kosten von X Ingenieurstunden."
- Setzen Sie eine messbare Schuldenobergrenze mit statischen Analysemetriken (maximales Verhältnis technischer Schulden, maximale Anzahl kritischer Code-Smells) und setzen Sie sie durch: Wenn die Obergrenze überschritten wird, hat Schuldenreduzierung Priorität vor neuen Features, bis sich die Metrik erholt.

## Tradeoffs ⇄

> Ein Backlog technischer Schulden macht das Unsichtbare sichtbar, was seine größte Stärke und die Quelle seines häufigsten Widerstands ist — Stakeholder, die die Schulden zuvor nicht sehen konnten, begrüßen es möglicherweise nicht, gezeigt zu bekommen, was sie schulden.

**Vorteile:**

- Verwandelt technische Schulden von einer unsichtbaren Bremse für Geschwindigkeit in ein verwaltetes Portfolio mit dokumentierten Kosten, was informierte Entscheidungen darüber ermöglicht, wann sie abzubauen sind versus wann sie zu akzeptieren sind.
- Liefert die Daten, die benötigt werden, um Modernisierungsinvestition gegenüber nicht-technischen Stakeholdern zu rechtfertigen, und ersetzt subjektive Beschwerden über "unordentlichen Code" durch konkrete Metriken und Geschäftsauswirkungsschätzungen.
- Verhindert die "Big-Bang-Neuschreibung"-Falle, indem inkrementelle, priorisierte Schuldenreduzierung ermöglicht wird — Teams, die Schulden kontinuierlich verwalten, vermeiden die Krise, die eine störende vollständige Neuschreibung erzwingt.
- Reduziert Produktionsvorfälle, indem brüchiger Code systematisch identifiziert und adressiert wird, bevor er versagt, statt ihn zu entdecken, wenn er einen Ausfall verursacht.
- Verbessert Entwicklerbindung und -moral, indem Teams ein Mechanismus gegeben wird, um ihre Arbeitsumgebung im Laufe der Zeit zu verbessern, statt unbegrenzte Verschlechterung als unvermeidlich zu akzeptieren.

**Kosten und Risiken:**

- Die Pflege des Schulden-Backlogs erfordert Disziplin und dedizierte Zeit; ein veraltetes Backlog, das den tatsächlichen Zustand der Codebasis nicht mehr widerspiegelt, schafft falsches Vertrauen und irrt Priorisierungsentscheidungen.
- In Legacy-Systemen mit massiv angesammelten Schulden kann das anfängliche Inventar so groß sein, dass es demoralisierend statt motivierend wirkt; das Backlog muss sofort priorisiert und abgegrenzt werden, um umsetzbar zu sein.
- Priorisierungsentscheidungen werden umstritten, wenn verschiedene Stakeholder unterschiedliche Ansichten darüber haben, was Auswirkung ausmacht; ohne ein gemeinsames Priorisierungs-Framework wird das Backlog zu einem politischen statt einem technischen Dokument.
- Statische Analysemetriken können perverse Anreize schaffen — Teams optimieren auf Schuldenverhältnis-Werte, indem sie Befunde unterdrücken oder Code oberflächlich umstrukturieren, ohne die zugrunde liegenden Qualitätsprobleme zu adressieren.
- Die Zuweisung von Sprint-Kapazität für Schuldenreduzierung reduziert kurzfristig den Feature-Durchsatz, was besonders sichtbar ist, wenn das Team unter Druck von Geschäfts-Stakeholdern steht, die den Zusammenhang zwischen Schulden und Liefergeschwindigkeit noch nicht sehen.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie Backlogs technischer Schulden in Legacy-Systemmodernisierungsprogrammen erstellt und genutzt werden.

Das Hypotheken-Origination-System einer nationalen Bank war seit achtzehn Jahren in Produktion gewesen. Das Entwicklungsteam verbrachte über die Hälfte seiner Zeit mit ungeplanter Wartung und Fehlerbehebungen, aber das Management schrieb die langsame Lieferung mangelndem Entwicklerengagement zu statt technischen Schulden. Das Team führte einen Schuldenentdeckungs-Sprint durch, unter Nutzung von SonarQube-Analyse kombiniert mit Entwickler-Schmerzpunkt-Interviews. Sie produzierten ein priorisiertes Backlog von vierzig Elementen und übersetzten die obersten zehn in Geschäftsbegriffe: zusammen machten sie geschätzt 35 % der ungeplanten Wartungszeit aus und hatten zu vier der sieben Produktionsvorfälle im vergangenen Jahr beigetragen. Mit diesem Backlog in der Hand genehmigte das Management eine ständige 25%ige Kapazitätszuweisung für Schuldenreduzierung — eine Verpflichtung, die in jedem vorherigen Gespräch abgelehnt worden war, in dem das Problem als "Code aufräumen" gerahmt wurde.

Eine von einem mittelgroßen Frachtunternehmen betriebene Logistikplattform war von einem Startup-Proof-of-Concept zu einem System herangewachsen, das täglich Zehntausende von Sendungen verarbeitete. Die Codebasis war nie refaktoriert worden, und die ursprünglichen Microservices hatten allmählich direkte Datenbankaufrufe über Servicegrenzen hinweg angesammelt, was die Isolation beseitigte, für die sie gestaltet waren. Das Team schuf ein Schulden-Backlog, das sich spezifisch auf Servicegrenzverletzungen konzentrierte, wobei Elemente nach den im aktuellen Quartal am häufigsten geänderten Diensten geordnet wurden. Durch die Reduzierung von Verletzungen in den drei aktivsten Diensten zuerst beseitigten sie eine Kategorie von Deployment-Tag-Fehlschlägen, die ein wiederkehrendes Problem gewesen war, und erweiterten erst dann ihre Anstrengung auf die weniger aktiven Teile des Systems.

Eine Regierungsbehörde, die ein Leistungsberechnungssystem in COBOL betrieb, musste eine zehnjährige Modernisierungs-Roadmap planen. Bevor Modernisierungsarbeit begann, stellte das Team einen externen Berater ein, um eine Schuldenbewertung durchzuführen, die automatisierte Komplexitätsanalyse mit strukturierten Interviews mit den zwei verbleibenden Entwicklern kombinierte, die institutionelles Wissen über das System hatten. Die Bewertung produzierte eine Modul-Ebenen-Schuldenkarte, die zeigte, welche Teile des Systems am höchsten riskant, komplexesten und geschäftskritischsten waren. Diese Karte wurde zum primären Input für die Modernisierungssequenzierungsentscheidung — das Team entschied sich, zuerst die Module mit den höchsten Schulden zu modernisieren, statt das System chronologisch abzuarbeiten, was das Risiko reduzierte, dass die Modernisierungsanstrengung technische Schulden von Modulen erbte, die bekanntermaßen brüchig waren.
