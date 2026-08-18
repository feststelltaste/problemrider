---
title: Implizites Wissen
description: Kritisches Systemwissen existiert als unausgesprochene Annahmen, Stammeswissen
  und undokumentierte Praktiken, statt explizit erfasst zu werden.
category:
- Communication
- Process
related_problems:
- slug: tacit-knowledge
  similarity: 0.7
- slug: knowledge-gaps
  similarity: 0.65
- slug: knowledge-dependency
  similarity: 0.65
- slug: knowledge-silos
  similarity: 0.65
- slug: legacy-system-documentation-archaeology
  similarity: 0.65
- slug: information-fragmentation
  similarity: 0.65
solutions:
- architecture-decision-records
- documentation-as-code
- knowledge-sharing-practices
- pair-and-mob-programming
- api-documentation
- architecture-documentation
- architecture-workshops
- business-process-modeling
- code-comments
- compatibility-requirements
- documentation-of-compatibility-requirements
- living-documentation
- raising-user-awareness
- security-community
- security-training
- domain-experts
- domain-quiz
- event-storming
- knowledge-base
- runbooks
- user-communities
- knowledge-rotation
- written-first-communication
layout: problem
lang: de
en_slug: implicit-knowledge
---

## Description

Implizites Wissen bezeichnet kritische Informationen über Systemverhalten, Geschäftsregeln, Implementierungsentscheidungen und operative Praktiken, die nur in den Köpfen erfahrener Teammitglieder existieren, statt explizit dokumentiert oder im Code erfasst zu sein. Dieses Wissen umfasst unausgesprochene Annahmen, kontextuelles Verständnis, historische Entscheidungen und praktisches Know-how, das essenziell ist, um das System zu verstehen und zu warten, aber nirgendwo formal festgehalten wird.

## Indicators ⟡

- Erfahrene Entwickler können schnell Probleme lösen, an denen Neulinge scheitern
- Systemverhalten hängt von unausgesprochenen Regeln und Annahmen ab
- Kritisches Wissen geht verloren, wenn erfahrene Teammitglieder das Unternehmen verlassen
- Neue Mitarbeiter stellen viele Fragen, die in bestehender Dokumentation nicht beantwortet werden
- Bestimmte Systemverhalten können nur von spezifischen Personen erklärt werden

## Symptoms ▲

- [Schwieriges Onboarding neuer Entwickler](schwieriges-onboarding-neuer-entwickler.md)
<br/>  Neue Mitarbeiter haben Schwierigkeiten, produktiv zu werden, weil kritisches Systemwissen nicht dokumentiert ist und durch Erfahrung oder Befragung von Personen gelernt werden muss.
- [Wissenssilos](wissenssilos.md)
<br/>  Wenn Wissen implizit ist, konzentriert es sich natürlich in den Köpfen spezifischer Personen, was gefährliche Einzelpunkte der Expertise schafft.
- [Hohe Rate an neu eingeführten Fehlern](hohe-rate-an-neu-eingefuehrten-fehlern.md)
<br/>  Entwickler, die sich unausgesprochener Regeln und Annahmen nicht bewusst sind, nehmen Änderungen vor, die implizite Einschränkungen verletzen, was Fehler einführt.
- [Implementierungs-Nacharbeit](implementierungs-nacharbeit.md)
<br/>  Features müssen neu gebaut werden, wenn Entwickler implizite Einschränkungen oder Geschäftsregeln entdecken, derer sie sich während der ursprünglichen Implementierung nicht bewusst waren.

## Causes ▼

- [Implizites Erfahrungswissen](implizites-erfahrungswissen.md)
<br/>  Wissen, das inhärent schwer zu artikulieren oder zu übertragen ist, wird natürlich implizit, statt in Dokumentation erfasst zu werden.
- [Schlechte Dokumentation](schlechte-dokumentation.md)
<br/>  Wenn Dokumentationspraktiken schlecht sind, bleibt Wissen, das aufgeschrieben werden sollte, nur in den Köpfen der Menschen.
- [Zeitdruck](zeitdruck.md)
<br/>  Unter Zeitdruck überspringen Teams Wissenserfassung und Dokumentation zugunsten schnellerer Feature-Lieferung.

## Detection Methods ○

- **Wissensabhängigkeits-Mapping:** Identifikation, welche Teammitglieder für bestimmte Arten von Problemen konsultiert werden
- **Analyse von Fragen neuer Mitarbeiter:** Nachverfolgung der Arten und Häufigkeit von Fragen neuer Teammitglieder
- **Bewertung von Dokumentationslücken:** Vergleich der Systemkomplexität mit der Umfassendheit schriftlicher Dokumentation
- **Auswirkung der Experten-Verfügbarkeit:** Messung, wie sehr das Systemverständnis leidet, wenn Schlüsselpersonen nicht verfügbar sind
- **Entscheidungsarchäologie:** Untersuchung, wie viele Systementscheidungen keine dokumentierte Begründung haben

## Examples

Ein Legacy-Finanzhandelssystem hat einen Konfigurationsparameter, der während Markt-Feiertagen auf einen bestimmten Wert gesetzt werden muss, aber diese Anforderung existiert nirgendwo in der Dokumentation. Nur der Senior-Architekt weiß, dass diese Einstellung eine Race Condition verhindert, die auftritt, wenn Marktdaten-Feeds während Feiertagsplänen inkonsistent sind. Als der Architekt in den Urlaub geht und ein Junior-Entwickler eine Konfigurationsänderung deployt, erlebt das System Datenverfälschungsprobleme, die Tage brauchen, um identifiziert und behoben zu werden. Ein weiteres Beispiel betrifft eine E-Commerce-Plattform, bei der die Bestellverarbeitungslogik subtile Timing-Abhängigkeiten hat, die erfordern, dass bestimmte Datenbankabfragen in einer bestimmten Reihenfolge ausgeführt werden. Dieses Wissen existiert nur in den Köpfen zweier Senior-Entwickler, die es durch jahrelange Fehlerbehebung in Produktion gelernt haben. Als das Team versucht, den Bestellverarbeitungscode zu optimieren, brechen sie unbeabsichtigt diese Timing-Annahmen und verursachen intermittierende Bestellfehler, die extrem schwer zu reproduzieren und zu debuggen sind.
