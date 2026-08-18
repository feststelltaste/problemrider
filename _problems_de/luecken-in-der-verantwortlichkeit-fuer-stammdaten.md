---
title: Lücken in der Verantwortlichkeit für Stammdaten
description: Zentrale Referenzdaten werden über Module und Abteilungen hinweg geteilt,
  ohne dass es einen Eigentümer gibt, sodass ihre Qualität sich verschlechtert und
  niemand für die Korrektur verantwortlich ist.
category:
- Database
- Management
- Business
related_problems:
- slug: lack-of-ownership-and-accountability
  similarity: 0.65
- slug: unclear-documentation-ownership
  similarity: 0.6
- slug: custom-report-sprawl
  similarity: 0.55
- slug: poor-interfaces-between-applications
  similarity: 0.5
- slug: poor-domain-model
  similarity: 0.5
- slug: information-fragmentation
  similarity: 0.5
solutions:
- master-data-stewardship
- clear-ownership-model
- data-quality-checks
- data-strategy
- canonical-data-model
- data-modeling
- consistent-terminology
- ubiquitous-language
- continuous-data-verification
- plausibility-checks
- data-deduplication
layout: problem
lang: de
en_slug: master-data-ownership-gaps
---

## Description

Lücken in der Verantwortlichkeit für Stammdaten treten auf, wenn die Referenzdaten, von denen viele Prozesse abhängen – Kunden, Lieferanten, Produkte, Kostenstellen, Organisationseinheiten – von mehreren Abteilungen erstellt und bearbeitet werden, ohne dass jemand für die Gesamtqualität verantwortlich ist. Kommerzielle Softwaresysteme machen dies durch ihr Design wahrscheinlich, weil ihre Module Stammdaten teilen und die Nutzer jedes Moduls die Felder pflegen, die ihnen wichtig sind. Das Ergebnis ist eine gemeinsam genutzte Ressource mit verteilter Pflege und ohne Verwalter: Doppeleinträge, die entstehen, weil die Suche schwieriger war als das Hinzufügen, Felder, die von denjenigen leer gelassen werden, die sie nicht benötigten, und inkonsistente Konventionen, die jede Abteilung für korrekt hält. Da keine einzelne Abteilung die vollen Kosten trägt, ist die Verschlechterung nur in nachgelagerten Symptomen sichtbar – fehlschlagende Schnittstellen, widersprüchliche Berichte, manueller Abgleich – die anderen Ursachen zugeschrieben werden.

## Indicators ⟡

- Derselbe Kunde, Lieferant oder dasselbe Produkt existiert mehrfach unter leicht unterschiedlichen Einträgen
- Abteilungen pflegen eigene Listen parallel zum System, weil der Version des Systems nicht vertraut werden kann
- Datenqualitätsprobleme werden wiederholt angesprochen, als Einzelkorrekturen behandelt und treten erneut auf
- Niemand kann sagen, wer berechtigt ist, einen Stammdatensatz zu erstellen, oder wer ihn genehmigt
- Benennungs- und Kodierungskonventionen unterscheiden sich je nach Abteilung, und jede hält ihre eigene für den Standard
- Schnittstellen zu anderen Systemen scheitern an Datensätzen, die in der Quelle gültig, aber nachgelagert unbrauchbar sind

## Symptoms ▲

- [Schattensysteme](schattensysteme.md)
<br/>  Abteilungen pflegen private Listen, weil sie sich nicht auf die gemeinsamen Daten verlassen können, und diese Listen werden tragend, ohne dass jemand dies entschieden hat.
- [Erhöhte manuelle Arbeit](erhoehte-manuelle-arbeit.md)
<br/>  Abgleich, Korrektur und Duplikat-Zusammenführung werden zu dauerhaften wiederkehrenden Aufgaben in mehreren Abteilungen gleichzeitig.
- [Schlechte Schnittstellen zwischen Anwendungen](schlechte-schnittstellen-zwischen-anwendungen.md)
<br/>  Nachgelagerte Systeme erhalten Datensätze, die intern gültig sind, aber Annahmen verletzen, die niemand dokumentiert hat, und Integrationen scheitern intermittierend.
- [Komplexität der Datenmigration](komplexitaet-der-datenmigration.md)
<br/>  Jede Migration muss zunächst die angehäuften Duplikate und Inkonsistenzen auflösen, was oft den größten Teil des Aufwands ausmacht.
- [Doppelter Aufwand](doppelter-aufwand.md)
<br/>  Mehrere Abteilungen pflegen, korrigieren und gleichen unabhängig voneinander überlappende Ansichten derselben Entitäten ab.
- [Wildwuchs an individuellen Reports](wildwuchs-an-individuellen-reports.md)
<br/>  Widersprüchliche Ausgaben vermehren sich, weil jede Abteilung über ihre eigene Interpretation der gemeinsamen Daten berichtet.

## Causes ▼

- [Fehlende Eigenverantwortung und Rechenschaftspflicht](fehlende-eigenverantwortung-und-rechenschaftspflicht.md)
<br/>  Gemeinsam genutzte Daten überschreiten Abteilungsgrenzen, und Organisationen weisen selten Eigentum an etwas zu, das dies tut.
- [Fehlpassung der Organisationsstruktur](fehlpassung-der-organisationsstruktur.md)
<br/>  Abteilungen sind um Funktionen herum organisiert, während die Daten um Entitäten herum organisiert sind, sodass kein Abteilungsauftrag den gesamten Lebenszyklus eines Datensatzes abdeckt.
- [Unzureichende Anforderungserhebung](unzureichende-anforderungserhebung.md)
<br/>  Stammdaten-Governance wird während der Implementierung selten als Anforderung formuliert, sodass nie ein Prozess dafür entworfen wird.
- [Übermäßige Anpassung](uebermaessige-anpassung.md)
<br/>  Abteilungsspezifische Felder und Validierungen häufen sich auf gemeinsamen Datensätzen an, und jeder Satz ist für eine Abteilung bedeutsam und für die anderen Rauschen.
- [Kurzfristiger Fokus](kurzfristiger-fokus.md)
<br/>  Das Erstellen eines Doppeleintrags dauert eine Minute und löst die heutige Aufgabe; die ordentliche Suche oder die Korrektur des zugrunde liegenden Eintrags nicht.

## Detection Methods ○

- Messung der Duplikatraten in den wichtigsten Stammdatenobjekten mittels Fuzzy-Matching auf Namen, Kennungen und Adressen
- Zählung von Datensätzen mit unvollständigen, in der Praxis obligatorischen Feldern und Prüfung, ob die Vollständigkeit je nach erstellender Abteilung variiert
- Fragen, wem Kundenstammdaten gehören, und Beobachtung, wie lange die Antwort dauert und wie viele Namen sie enthält
- Nachverfolgung, wie viel Aufwand in Datenkorrektur und -abgleich über Abteilungen hinweg fließt, was üblicherweise unerfasst und erheblich ist
- Überprüfung von Schnittstellenfehlerprotokollen auf Ablehnungen, die durch Quelldaten statt technische Fehler verursacht werden
- Suche nach Abteilungs-Tabellenkalkulationen, die Stammdaten duplizieren, was darauf hinweist, wo Vertrauen bereits verloren gegangen ist

## Examples

Der Lieferantenstamm eines Herstellers enthielt 41.000 Datensätze, von denen eine Analyse mittels Fuzzy-Matching auf Namen, Steuernummer und Bankdaten etwa 6.800 als wahrscheinliche Duplikate identifizierte. Der Einkauf erstellte Datensätze zur Abwicklung einer Bestellung, die Finanzabteilung erstellte sie zur Verarbeitung einer Rechnung, und keine suchte gründlich, weil die Suche langsamer war als das Erstellen. Die Konsequenzen zeigten sich an anderer Stelle: Die Ausgabenanalyse unterschätzte die Konzentration bei einzelnen Lieferanten, was dazu geführt hatte, dass eine Verhandlung aus einer schwächeren Position eröffnet wurde, als die Fakten hergaben, und ein Zahlungslauf hatte zweimal Doppelzahlungen an denselben Lieferanten unter zwei Datensätzen gesendet.

Die Eigentumsfrage erwies sich als das eigentliche Problem. Auf die Frage, wem die Lieferantenstammdaten gehörten, lieferte die Organisation drei Antworten – Einkauf, Finanzen und das Team des Unternehmenssystems –, wobei jede ihre Rolle als Pflege eines Teilbereichs betrachtete. Niemand war dafür verantwortlich, ob das Objekt als Ganzes korrekt war. Die Intervention war nicht technischer Natur: ein benannter Verwalter pro Stammdatenobjekt, ein definierter Erstellungsprozess mit obligatorischem Suchschritt und ein monatlicher Duplikat-Report, der an den Verwalter statt an eine Verteilerliste gesendet wird. Die Duplikaterstellung sank innerhalb von zwei Quartalen um etwa neunzig Prozent, und der Bestand an bestehenden Duplikaten wurde zu einer begrenzten Arbeit mit einem Eigentümer statt zu einem dauerhaften Zustand.
