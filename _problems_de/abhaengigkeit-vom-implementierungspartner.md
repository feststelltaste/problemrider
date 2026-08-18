---
title: Abhängigkeit vom Implementierungspartner
description: Nur die externe Beratungsfirma versteht, wie das System gebaut wurde,
  sodass die Organisation ihre eigene Installation ohne sie nicht ändern, bewerten
  oder verlassen kann.
category:
- Dependencies
- Team
- Management
related_problems:
- slug: dependency-on-supplier
  similarity: 0.7
- slug: vendor-dependency
  similarity: 0.7
- slug: vendor-dependency-entrapment
  similarity: 0.7
- slug: knowledge-dependency
  similarity: 0.65
- slug: vendor-lock-in
  similarity: 0.65
- slug: reimplemented-standard-functionality
  similarity: 0.6
solutions:
- vendor-management-practice
- knowledge-rotation
- internal-technical-coaching
- code-reading-sessions
- documentation-as-code
- architecture-decision-records
- customization-under-version-control
- pair-and-mob-programming
- technical-skills-development
- risk-quantification
- structured-onboarding-program
layout: problem
lang: de
en_slug: implementation-partner-dependency
---

## Description

Abhängigkeit vom Implementierungspartner entsteht, wenn das Wissen darüber, wie ein kommerziell erworbenes Softwaresystem konfiguriert, erweitert und integriert wurde, bei einer externen Beratungsfirma liegt statt innerhalb der Organisation, die es besitzt. Sie entwickelt sich natürlich: Der Partner implementiert, interne Mitarbeiter betreiben, und das Verständnis dafür, warum die Dinge so sind, wie sie sind, geht nie über, weil niemand dafür sorgt. Die Abhängigkeit ist einschränkender als gewöhnliche Zulieferabhängigkeit, weil sie die eigene Konfiguration der Organisation betrifft statt ein Produkt. Der Partner kann nur durch einen anderen Partner ersetzt werden, der bereit ist, Monate damit zu verbringen, zu lernen, was der erste weiß, und diese Kosten machen die Sätze des Amtsinhabers effektiv unangreifbar. Organisationen erkennen die Situation häufig erst, wenn sie versuchen, Partner zu wechseln oder Arbeit intern zu übernehmen, und feststellen, dass beides nicht verfügbar ist.

## Indicators ⟡

- Jede nicht-triviale Änderung erfordert den Partner, einschließlich Änderungen, die als Konfiguration erscheinen
- Interne Mitarbeiter können das System bedienen, aber nicht erklären, warum es sich so verhält, wie es sich verhält
- Schätzungen des Partners können nicht unabhängig bewertet werden und werden akzeptiert, weil es keine Grundlage gibt, sie anzuzweifeln
- Die Dokumentation der Implementierung ist dünn, veraltet oder besteht aus den ursprünglichen Designdokumenten des Partners
- Dieselben einzelnen Berater arbeiten seit Jahren am Konto, und ihre Abwesenheit ist ein Terminplanungsrisiko
- Das Einholen eines konkurrierenden Angebots wird als unpraktisch betrachtet, weil ein Wettbewerber das System erst lernen müsste

## Symptoms ▲

- [Wissensabhängigkeit](wissensabhaengigkeit.md)
<br/>  Kritisches Verständnis des eigenen Systems der Organisation liegt außerhalb von ihr und kann nur über eine kommerzielle Vereinbarung abgerufen werden.
- [Hohe Wartungskosten](hohe-wartungskosten.md)
<br/>  Sätze können nicht gegen Alternativen getestet werden, und Arbeit, die interne Mitarbeiter leisten könnten, muss stattdessen eingekauft werden.
- [Langsame Feature-Entwicklung](langsame-feature-entwicklung.md)
<br/>  Änderungen bewegen sich in der Geschwindigkeit der Verfügbarkeit und des Vertragszyklus des Partners statt im Bedürfnis der Organisation.
- [Belastete Anbieterbeziehung](belastete-anbieterbeziehung.md)
<br/>  Eine unausgewogene Abhängigkeit erzeugt Groll auf der Kundenseite und Selbstgefälligkeit auf der Zuliefererseite, und beides zeigt sich in der Beziehung.
- [Lähmung der Modernisierungsstrategie](laehmung-der-modernisierungsstrategie.md)
<br/>  Die Bewertung von Optionen erfordert das Verständnis der aktuellen Installation, was die Organisation nicht leisten kann, ohne die Partei zu fragen, die ein Interesse an der Antwort hat.
- [Schlechte Dokumentation](schlechte-dokumentation.md)
<br/>  Dokumentation ist ein Liefergegenstand, für den der Partner bezahlt wird und den zu pflegen keine betriebliche Notwendigkeit für ihn besteht, sodass sie schnell nicht mehr der Realität entspricht.

## Causes ▼

- [Abhängigkeit vom Zulieferer](abhaengigkeit-vom-zulieferer.md)
<br/>  Die kommerzielle Beziehung wurde um Lieferung statt um Fähigkeitstransfer herum strukturiert, und nichts darin erforderte, dass Wissen sich bewegt.
- [Mangel an Legacy-Fachkräften](mangel-an-legacy-fachkraeften.md)
<br/>  Ohne interne Personen, die das Wissen hätten aufnehmen können, gab es niemanden, an den der Partner hätte übergeben können, selbst wo Übergabe beabsichtigt war.
- [Probleme mit der Personalverfügbarkeit](probleme-mit-der-personalverfuegbarkeit.md)
<br/>  Interne Mitarbeiter sind vollständig für den Betrieb eingeplant, sodass die Teilnahme an Implementierungsarbeit aufgeschoben wird und der Transfer nie stattfindet.
- [Kurzfristiger Fokus](kurzfristiger-fokus.md)
<br/>  Die Arbeit einzukaufen ist schneller, als die Fähigkeit aufzubauen, und dieser Vergleich wird bei jeder Gelegenheit mit demselben Ergebnis angestellt.
- [Schlechtes Vertragsdesign](schlechtes-vertragsdesign.md)
<br/>  Verträge spezifizieren Liefergegenstände statt Wissenstransfer, Dokumentationsstandards oder die Fähigkeit des Kunden, ohne den Zulieferer fortzufahren.
- [Hohe Fluktuation](hohe-fluktuation.md)
<br/>  Internes Wissen, das sich entwickelte, geht mit den Menschen, die es hielten, während das Kontoteam des Partners stabil bleibt, was die Asymmetrie vergrößert.

## Detection Methods ○

- Fragen, welcher Anteil der Änderungen im letzten Jahr den Partner erforderte und wie viele davon Konfiguration statt Entwicklung waren
- Feststellen, ob irgendjemand intern die drei wichtigsten Anpassungen erklären könnte, ohne den Partner zu konsultieren
- Schätzen, was ein konkurrierender Zulieferer ausgeben müsste, um beim Konto produktiv zu werden; diese Zahl ist die Wechselkosten
- Prüfen, ob der Vertrag eine Verpflichtung bezüglich Dokumentationsqualität oder Wissenstransfer enthält, und ob sie durchgesetzt wurde
- Die Position testen: einer internen Person eine echte Änderung zuweisen und messen, was sie ohne Hilfe erreichen kann
- Überprüfen, ob Schätzungen des Partners je erfolgreich angezweifelt wurden und auf welcher Grundlage

## Examples

Ein regionaler Versorger betrieb neun Jahre lang eine Enterprise-Resource-Planning-Installation, die durchgängig von einer Beratungsfirma implementiert und gewartet wurde. Als die Beschaffungsabteilung Wettbewerbsangebote für eine geplante Erweiterung suchte, lehnten zwei Zulieferer ab zu bieten, und der dritte nannte einen Betrag, der vier Monate Einarbeitung einschloss. Das Angebot des Amtsinhabers war niedriger und wurde akzeptiert, wie es seit neun Jahren der Fall war. Eine interne Überprüfung stellte anschließend fest, dass kein Mitarbeiter beschreiben konnte, wie die eigenen Preisregeln der Organisation implementiert waren, dass die Designdokumentation aus der ursprünglichen Implementierung stammte und dass drei der Berater des Partners effektiv das gesamte operative Wissen über die Installation hielten.

Die Reaktion der Organisation ist aufschlussreich, weil die naheliegende nicht funktioniert hätte. Sie versuchten nicht, den Partner zu ersetzen, was die Wechselkosten sofort realisiert hätte. Stattdessen fügten sie den nächsten drei Änderungsprojekten zwei interne Personen als Teilnehmer statt als Beobachter hinzu, verlangten, dass die Arbeit des Partners in einem internen Repository mit aufgezeichnetem Design landete, und stellten eine Regel auf, dass der Partner nicht die einzige Partei sein durfte, die einen Bereich berührt hatte. Nach achtzehn Monaten handhabte das interne Team etwa ein Drittel der Änderungen ohne Hilfe, und das nächste Wettbewerbsangebot zog drei Gebote an – nicht weil das System einfacher geworden war, sondern weil es für einen Außenstehenden beschreibbar geworden war.
