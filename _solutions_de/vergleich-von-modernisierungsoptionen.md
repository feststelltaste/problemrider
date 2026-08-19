---
title: Vergleich von Modernisierungsoptionen
description: Abschalten, Beibehalten, Kapseln, Ersetzen und Neuschreiben als
  bepreiste Alternativen nebeneinander präsentieren, statt um Genehmigung für
  eine bevorzugte Antwort zu bitten.
category:
- Architecture
- Management
- Business
problems:
- modernization-roi-justification-failure
- modernization-strategy-paralysis
- difficulty-quantifying-benefits
- second-system-effect
- technology-lock-in
- obsolete-technologies
- system-stagnation
- decision-paralysis
- budget-overruns
- high-maintenance-costs
- accumulated-decision-debt
- premature-technology-introduction
- competitive-disadvantage
- legacy-skill-shortage
- technology-stack-fragmentation
- vendor-dependency
- vendor-dependency-entrapment
- core-modification-of-standard-software
- upgrade-blocked-by-customization
layout: solution
lang: de
en_slug: modernization-options-comparison
related_solutions:
- slug: no-regret-moves
  similarity: 0.75
- slug: strangler-fig-pattern
  similarity: 0.7
- slug: total-cost-of-ownership-transparency
  similarity: 0.7
- slug: boring-technologies
  similarity: 0.7
- slug: cost-of-delay
  similarity: 0.7
- slug: risk-analysis
  similarity: 0.65
---

## Description

Ein Vergleich von Modernisierungsoptionen präsentiert die realistischen Alternativen für ein Legacy-System — es abschalten, es beibehalten wie es ist, es hinter einer Schnittstelle kapseln, es durch ein Produkt ersetzen, es neu schreiben oder es weitgehend unverändert migrieren — als bepreiste, risikobewertete Alternativen, bewertet anhand derselben Kriterien. Die übliche Praxis ist anders: Ein Team entscheidet intern, welche Option es bevorzugt, baut einen Fall für diese eine auf und präsentiert ihn zur Genehmigung. Diese Rahmung versetzt den Entscheider in die Position, einen einzigen Vorschlag entweder zu akzeptieren oder abzulehnen, ohne Grundlage zu beurteilen, ob die bevorzugte Option die richtige ist. Sie entzieht auch deren Handlungsspielraum, was zuverlässig entweder Ablehnung oder eine Forderung nach mehr Analyse produziert. Alternativen zu präsentieren verändert das Gespräch von der Frage, ob die Antwort des Engineerings genehmigt werden soll, zu der Frage, welchen Tradeoff die Organisation eingehen möchte — eine Entscheidung, für die die Organisation tatsächlich ausgestattet ist.

## How to Apply ◆

> Zwei der Optionen werden systematisch unterexaminiert: das System vollständig abzuschalten und es bewusst zu behalten, wie es ist — und eine davon ist häufig die richtige.

- **Beziehen Sie Abschalten und Beibehalten als echte Kandidaten ein**, bewertet mit derselben Ernsthaftigkeit wie die anderen. Nutzungsdaten zeigen gelegentlich, dass ein System weit weniger bedient, als angenommen, und „behalten, eindämmen und das Geld anderswo ausgeben" ist ein legitimes Ergebnis, das ein Vergleich erreichen kann und ein Einzeloptions-Vorschlag nie kann.
- **Nutzen Sie dieselben Kriterien für jede Option**: Kosten zum Erreichen des Endzustands, resultierende Betriebskosten, Risiko während des Übergangs, Risiko des Endzustands, Zeit bis zum ersten Nutzen und was ausgeschlossen wird. Anhand unterschiedlicher Kriterien bewertete Optionen können nicht verglichen werden, und eine inkonsistente Tabelle ist der schnellste Weg, wie Advocacy auszusehen.
- **Bepreisen Sie auch die Nichts-tun-Option.** Weitermachen wie bisher ist nicht kostenlos, und drei Änderungsoptionen gegen eine implizite Null zu vergleichen ist der häufigste Fehler in diesen Dokumenten. Die Kosten der Verzögerung für den aktuellen Zustand sind hier die richtige Zahl.
- **Erklären Sie die Zuversicht jeder Schätzung**, nicht nur die Zahl. Eine Neuschreibungs-Schätzung verdient eine weit größere Spanne als eine Kapselungs-Schätzung, und diesen Unterschied hinter zwei gleich präzise aussehenden Zahlen zu verstecken führt den Leser über das echte Risiko in die Irre.
- **Bewerten Sie die Zeit bis zum ersten Nutzen separat von den Gesamtkosten.** Eine Option, die mehr kostet, aber in vier Monaten etwas liefert, ist in einer Organisation, die die Ausgabe jährlich verteidigen muss, oft vorzuziehen, und diese Dimension wird routinemäßig weggelassen.
- **Vermerken Sie, was jede Option ausschließt.** Kapselung bewahrt die Option, später zu ersetzen; eine Neuschreibung verpflichtet. Optionalität hat echten Wert unter Unsicherheit und gehört explizit in den Vergleich.
- **Geben Sie eine Empfehlung, mit Begründung.** Ein Vergleich, der sich weigert zu empfehlen, verzichtet auf die Expertise, um die das Team gebeten wurde. Der Punkt ist, dass die Empfehlung sichtbar aus dem Vergleich abgeleitet ist statt ihm vorauszugehen.
- **Zeigen Sie das Hybrid.** Die realistische Antwort für einen großen Legacy-Bestand ist meist verschiedene Optionen für verschiedene Teile — zwei Module abschalten, drei kapseln, eines ersetzen. Nur Ganzsystemoptionen zu präsentieren stellt die tatsächliche Wahl falsch dar.
- **Lassen Sie die Schätzungen von jemandem ohne Eigeninteresse überprüfen**, vor der Präsentation. Der häufigste Fehler ist, dass die Schätzung der bevorzugten Option optimistisch ist und die der Alternativen nicht, was selten absichtlich ist und immer entdeckt wird.

## Tradeoffs ⇄

> Der Vergleich von Optionen produziert bessere Entscheidungen und weit bessere Glaubwürdigkeit, auf Kosten des Schätzaufwands für Pfade, die nicht eingeschlagen werden.

**Vorteile:**

- Der Entscheider kann Tradeoffs abwägen, statt einen einzigen Vorschlag zu akzeptieren oder abzulehnen, was sowohl ein besserer Entscheidungsprozess ist als auch weit wahrscheinlicher in Genehmigung endet.
- Die Glaubwürdigkeit des Teams steigt erheblich, weil ein Dokument, das Alternativen zu seiner eigenen Empfehlung ernsthaft bewertet, nicht wie Advocacy wirkt.
- Abschalten und Beibehalten erhalten echte Berücksichtigung, und eine davon ist häufiger die richtige Antwort, als engineering-geführte Vorschläge nahelegen.
- Der Vergleich bringt Hybride ans Licht, die meist die realistische Antwort für ein System jeder Größe sind.
- Wenn sich die gewählte Option später als schwierig erweist, verhindert die Aufzeichnung dessen, was verglichen wurde und warum, dass die Entscheidung von Grund auf neu verhandelt wird.

**Kosten und Risiken:**

- Optionen zu schätzen, die nicht verfolgt werden, ist echter Aufwand ohne direkten Ertrag, und er verzögert die Entscheidung.
- Die Schätzqualität variiert enorm über Optionen hinweg, und eine gut fundierte Kapselungszahl neben eine spekulative Neuschreibungszahl zu stellen verleiht beiden einen falschen Anschein von Vergleichbarkeit.
- Mehr Optionen können Paralyse vertiefen statt sie zu lösen, besonders in Organisationen, die bereits zu Aufschub neigen.
- Der Vergleich kann, bewusst oder nicht, so konstruiert werden, dass die bevorzugte Option gewinnt — durch Kriterienauswahl ebenso sehr wie durch Schätzungen.
- Eine echte Option zu präsentieren, die das Team für falsch hält, riskiert, dass sie gewählt wird, was ein echter Preis der von der Methode verlangten Ehrlichkeit ist.

## How It Could Be

Das Lagerverwaltungssystem eines Logistikunternehmens war Gegenstand eines Neuschreibungsvorschlags, geschätzt auf 6 Millionen Euro über zwei Jahre. Der Vorstand lehnte ihn zweimal ohne erklärten Grund ab. Beim dritten Versuch präsentierte das Team fünf bepreiste Optionen statt einer. Weitermachen wie bisher kam mit Verzögerungskosten von etwa 95.000 Euro pro Monat, steigend. Kapselung hinter einer Serviceschicht: 1,4 Millionen Euro, achtzehn Monate bis zum Abschluss, erster Nutzen in vier Monaten, bewahrt die Option, später zu ersetzen. Paketersatz: 3,2 Millionen Euro mit weiter Spanne, hohes Übergangsrisiko, eine Passungsbewertung, die zeigte, dass zwei von elf erforderlichen Fähigkeiten nicht unterstützt wurden. Neuschreibung: 6,1 Millionen Euro mit sehr weiter Spanne, erster Nutzen in etwa zwanzig Monaten. Abschalten und in das andere Lagersystem der Gruppe absorbieren: 2,1 Millionen Euro, aber eine geschäftliche Entscheidung über Standortkonsolidierung erfordernd, die nicht Sache des Engineerings war. Der Vorstand wählte Kapselung und eröffnete separat die Konsolidierungsfrage, die die Abschaltoption zutage gebracht hatte.

Das wertvollste Ergebnis des Vergleichs war eines, das niemand einzubeziehen erwartet hatte. Die Bewertung der Abschaltoption erforderte, dass jemand fragte, welche Standorte das System tatsächlich bediente, was den Befund produzierte, dass zwei der sieben drei Jahre zuvor zum Gruppensystem migriert worden waren und dass das Legacy-System immer noch Schnittstellen für sie betrieb — gepflegt, überwacht, gepatcht und von nichts genutzt. Diese Schnittstellen wurden innerhalb eines Monats stillgelegt, unabhängig davon, welche Option gewählt wurde. Die Schätzung des Teams für die wiederkehrende Einsparung betrug etwa 140.000 Euro pro Jahr, gefunden als Nebeneffekt davon, eine Option ernst zu nehmen, die sie für nicht tragfähig gehalten hatten.
