---
title: Informationsfragmentierung
description: Kritisches Systemwissen ist über mehrere Orte und Formate verstreut,
  was es schwierig macht, es zu finden und zu pflegen.
category:
- Communication
- Process
related_problems:
- slug: knowledge-sharing-breakdown
  similarity: 0.7
- slug: knowledge-gaps
  similarity: 0.7
- slug: knowledge-silos
  similarity: 0.7
- slug: information-decay
  similarity: 0.65
- slug: technology-stack-fragmentation
  similarity: 0.65
- slug: implicit-knowledge
  similarity: 0.65
solutions:
- documentation-as-code
- data-integration
- knowledge-base
- living-documentation
- architecture-documentation
- consistent-terminology
- search-function
- written-first-communication
layout: problem
lang: de
en_slug: information-fragmentation
---

## Description

Informationsfragmentierung tritt auf, wenn kritisches Systemwissen, Entscheidungen und Dokumentation über mehrere unverbundene Orte, Formate und Systeme verstreut sind. Dies schafft eine Situation, in der Teammitglieder die benötigten Informationen nicht effizient auffinden können, was zu duplizierten Rechercheanstrengungen, inkonsistenter Entscheidungsfindung und Wissensverlust führt. Anders als das völlige Fehlen von Dokumentation existieren fragmentierte Informationen, sind aber aufgrund schlechter Organisation und Auffindbarkeit effektiv unzugänglich.

## Indicators ⟡

- Teammitglieder fragen häufig "Wo finde ich Informationen über..."
- Ähnliche Fragen werden wiederholt gestellt, weil vorherige Antworten schwer zu finden sind
- Dokumentation existiert in mehreren Formaten über unterschiedliche Systeme hinweg (Wikis, gemeinsame Laufwerke, E-Mails, Chat-Historie)
- Die Suchfunktionalität über Informationsquellen hinweg ist schlecht oder nicht vorhanden
- Kritische Entscheidungen und ihre Begründung sind in Meeting-Notizen oder Chat-Gesprächen vergraben

## Symptoms ▲

- [Wissenssilos](wissenssilos.md)
<br/>  Wenn Informationen verstreut sind, können nur diejenigen, die wissen, wo sie suchen müssen, sie finden, was De-facto-Wissenssilos schafft.
- [Schwieriges Onboarding neuer Entwickler](schwieriges-onboarding-neuer-entwickler.md)
<br/>  Neue Teammitglieder haben Schwierigkeiten, die benötigten Informationen zu finden, wenn sie über mehrere unverbundene Systeme verstreut sind.
- [Verschwendeter Entwicklungsaufwand](verschwendeter-entwicklungsaufwand.md)
<br/>  Entwickler verschwenden Zeit mit der Suche nach Informationen oder der Duplizierung von Recherche, die bereits gemacht wurde, aber an einem unauffindbaren Ort gespeichert ist.
- [Verzögerte Entscheidungsfindung](verzoegerte-entscheidungsfindung.md)
<br/>  Wenn Teammitglieder frühere Entscheidungen nicht finden können, treffen sie neue Entscheidungen, die früheren widersprechen könnten.
- [Informationsverfall](informationsverfall.md)
<br/>  Fragmentierte Informationen sind schwerer zu pflegen und zu aktualisieren, was ihren Verfall in Ungenauigkeit beschleunigt.

## Causes ▼

- [Schlechte Dokumentation](schlechte-dokumentation.md)
<br/>  Fehlende Dokumentationsstandards und -praktiken führen dazu, dass Informationen inkonsistent über mehrere Orte hinweg festgehalten werden.
- [Werkzeugeinschränkungen](werkzeugeinschraenkungen.md)
<br/>  Die Nutzung vieler unterschiedlicher Werkzeuge für Dokumentation und Kommunikation verstreut Informationen natürlich über Systeme hinweg.

## Detection Methods ○

- **Informations-Audit:** Befragung, welche kritischen Informationen existieren und wo sie sich befinden
- **Test der Suchwirksamkeit:** Messung, wie lange Teammitglieder brauchen, um bestimmte Informationen zu finden
- **Fragemuster-Analyse:** Nachverfolgung häufig wiederholter Fragen, die auf Probleme beim Auffinden von Informationen hindeuten
- **Werkzeugnutzungsanalyse:** Kartierung, welche Informationssysteme genutzt werden und wie sie verbunden sind
- **Erfahrung neuer Teammitglieder:** Beobachtung, wie effektiv neue Mitarbeiter notwendige Informationen finden können

## Examples

Ein Entwicklungsteam hat kritische API-Dokumentation an drei unterschiedlichen Orten: ursprüngliche Spezifikationen in Google Drive, Implementierungsnotizen in Confluence und Fehlerbehebungstipps über Slack-Gespräche verstreut. Wenn ein neuer Entwickler die API integrieren muss, verbringt er zwei Tage mit der Suche durch diese Quellen und übersieht immer noch wichtige Implementierungsdetails, die vor sechs Monaten in einem Slack-Thread diskutiert wurden. Ein weiteres Beispiel betrifft ein Team, bei dem architektonische Entscheidungen in Meeting-Notizen dokumentiert sind, die in verschiedenen Ordnern gespeichert sind, wobei manche Entscheidungen in JIRA-Kommentaren, andere in Wiki-Seiten und wieder andere nur in E-Mail-Threads festgehalten sind. Wenn sie verstehen müssen, warum eine bestimmte Technologieentscheidung getroffen wurde, müssen Teammitglieder mehrere Systeme durchsuchen und finden oft nicht die vollständige Begründung, was zu wiederholten architektonischen Diskussionen führt.
