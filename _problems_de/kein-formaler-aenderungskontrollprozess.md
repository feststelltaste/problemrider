---
title: Kein formaler Änderungskontrollprozess
description: Änderungen am Projektumfang oder an Anforderungen werden nicht formal
  bewertet oder genehmigt, was zu unkontrolliertem Scope Creep und Projektverzögerungen
  führt.
category:
- Process
related_problems:
- slug: scope-creep
  similarity: 0.75
- slug: changing-project-scope
  similarity: 0.7
- slug: frequent-changes-to-requirements
  similarity: 0.7
- slug: change-management-chaos
  similarity: 0.7
- slug: scope-change-resistance
  similarity: 0.7
- slug: poor-project-control
  similarity: 0.65
solutions:
- change-management-process
- formal-change-control-process
- product-owner
- version-control
- decision-rights-and-escalation
- change-impact-analysis
- definition-of-done
- runbooks
- production-readiness-criteria
layout: problem
lang: de
en_slug: no-formal-change-control-process
---

## Description
Ein formaler Änderungskontrollprozess ist essenziell für die Verwaltung der Entwicklung des Umfangs und der Anforderungen eines Projekts. Ohne einen solchen sind Projekte anfällig für Scope Creep, bei dem neue Features und Änderungen ohne ordentliche Bewertung ihrer Auswirkung auf Zeitpläne, Budgets oder Ressourcen hinzugefügt werden. Dies kann zu einem chaotischen Entwicklungsprozess, verpassten Terminen und einem finalen Produkt führen, das nicht mit der ursprünglichen Vision übereinstimmt. Ein Mangel an formaler Änderungskontrolle entsteht oft aus dem Wunsch, flexibel und reaktionsfähig zu sein, untergräbt aber letztlich die Stabilität und den Erfolg des Projekts.

## Indicators ⟡
- Der Umfang des Projekts erweitert sich ständig.
- Das Team verpasst häufig Termine.
- Das Team wechselt ständig den Kontext.
- Es gibt viel Nacharbeit.

## Symptoms ▲

- [Scope Creep](scope-creep.md)
<br/>  Ohne formale Bewertung von Änderungen werden neue Anfragen kontinuierlich hinzugefügt, ohne ihre Auswirkung zu bewerten, was unkontrollierte Umfangserweiterung verursacht.
- [Verpasste Termine](verpasste-termine.md)
<br/>  Unbewertete Änderungen verbrauchen Entwicklungskapazität, die für geplante Arbeit vorgesehen war, was Terminverzug verursacht.
- [Overhead durch Kontextwechsel](overhead-durch-kontextwechsel.md)
<br/>  Ad-hoc-Änderungsanfragen unterbrechen den Fokus der Entwickler, während sie zwischen geplanter Arbeit und unverwalteten Anfragen hin- und hergerissen werden.
- [Ressourcenverschwendung](ressourcenverschwendung.md)
<br/>  Erhebliche Nacharbeit entsteht, wenn unkontrollierte Änderungen miteinander in Konflikt geraten oder zuvor abgeschlossene Arbeit ungültig machen.
- [Entwicklerfrustration und Burnout](entwicklerfrustration-und-burnout.md)
<br/>  Ständig verschiebende Prioritäten und wachsende Backlogs durch unverwaltete Änderungen führen zu Erschöpfung und Frustration im Team.
- [Übermäßige Gefälligkeit gegenüber Stakeholdern](uebermaessige-gefaelligkeit-gegenueber-stakeholdern.md)
<br/>  Ohne formale Änderungsbewertungsprozesse verfallen Teams standardmäßig darauf, jeder Stakeholder-Anfrage zuzustimmen, um Konflikt zu vermeiden.

## Causes ▼

- [Schlechte Projektsteuerung](schlechte-projektsteuerung.md)
<br/>  Schwache Projekt-Governance-Strukturen versäumen es, formale Prozesse zur Verwaltung von Änderungen zu etablieren und durchzusetzen.
- [Ineffiziente Prozesse](ineffiziente-prozesse.md)
<br/>  Organisationen mit unausgereiften Entwicklungsprozessen fehlt oft die Disziplin, formale Änderungskontrolle zu implementieren und zu befolgen.
- [Schlecht definierte Verantwortlichkeiten](schlecht-definierte-verantwortlichkeiten.md)
<br/>  Wenn niemand klar dafür verantwortlich ist, Änderungen zu genehmigen oder abzulehnen, fließen alle Anfragen direkt und ungeprüft zum Entwicklungsteam.

## Detection Methods ○

- **Projekt-Audits:** Überprüfung von Projektdokumentation, Meeting-Protokollen und Kommunikationsprotokollen, um zu sehen, wie Änderungen verwaltet werden.
- **Vergleich von Baselines:** Regelmäßiger Vergleich des aktuellen Projektumfangs und -plans mit der ursprünglichen Baseline zur Identifikation unverwalteter Abweichungen.
- **Stakeholder-Interviews:** Befragung von Stakeholdern und Teammitgliedern zu ihrer Erfahrung mit der Verwaltung von Änderungen und ihrem Verständnis des Prozesses.
- **Nachverfolgung von Nacharbeitsmetriken:** Überwachung des Entwicklungsaufwands, der für die Neuimplementierung oder Modifikation bereits abgeschlossener Features aufgewendet wird.

## Examples
Ein Softwareentwicklungsprojekt nähert sich seinem Release-Termin. Ein wichtiger Geschäfts-Stakeholder erwähnt beiläufig in einem Flurgespräch, dass ein kritischer neuer Bericht vor dem Launch benötigt wird. Ohne formalen Änderungskontrollprozess wird diese Anfrage sofort zum Entwicklungs-Backlog hinzugefügt, was eine erhebliche Verzögerung des Releases verursacht und andere geplante Features beeinträchtigt. In einem anderen Fall baut ein Team eine mobile Anwendung. Über mehrere Monate senden verschiedene Produktmanager und Designer einzelne E-Mails mit neuen Feature-Ideen oder Modifikationen. Ohne ein zentralisiertes System zur Nachverfolgung und Genehmigung dieser wird das Entwicklungsteam überfordert, und das Projekt gerät mit einer ständig wachsenden Liste unpriorisierter Features hinter den Zeitplan zurück. Dieses Problem ist eine häufige Falle im Projektmanagement, besonders in Organisationen, denen es an Reife in ihrem Softwareentwicklungszyklus mangelt. Es trägt direkt zu Projektfehlschlägen, Budgetüberschreitungen und Team-Burnout bei und ist besonders herausfordernd bei Legacy-Modernisierungsbemühungen, wo der Umfang inhärent fließend sein kann.
