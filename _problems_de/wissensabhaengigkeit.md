---
title: Wissensabhängigkeit
description: Teammitglieder bleiben länger als für ihre Rolle und Betriebszugehörigkeit
  angemessen von bestimmten erfahrenen Personen für Wissen und Entscheidungsfindung
  abhängig.
category:
- Communication
- Dependencies
- Team
related_problems:
- slug: knowledge-silos
  similarity: 0.75
- slug: knowledge-gaps
  similarity: 0.7
- slug: single-points-of-failure
  similarity: 0.7
- slug: approval-dependencies
  similarity: 0.7
- slug: slow-knowledge-transfer
  similarity: 0.65
- slug: implicit-knowledge
  similarity: 0.65
solutions:
- knowledge-sharing-practices
- pair-and-mob-programming
- collaborative-problem-solving
- runbooks
- knowledge-rotation
- code-reading-sessions
- internal-technical-coaching
- documentation-as-code
- knowledge-base
layout: problem
lang: de
en_slug: knowledge-dependency
---

## Description

Wissensabhängigkeit tritt auf, wenn sich Teammitglieder, besonders solche, die keine neuen Mitarbeiter mehr sind, weiterhin stark auf bestimmte erfahrene Personen für Informationen, Entscheidungen und Anleitung verlassen, die sie vernünftigerweise selbstständig handhaben können sollten. Dies schafft eine Situation, in der Teammitglieder nicht autonom arbeiten können und erfahrene Entwickler zu Engpässen für Routineaufgaben und Entscheidungen werden.

## Indicators ⟡

- Entwickler mit Monaten oder Jahren an Betriebszugehörigkeit stellen immer noch grundlegende Fragen zur Systemfunktionalität
- Teammitglieder warten darauf, dass bestimmte Personen verfügbar sind, bevor sie mit Aufgaben fortfahren
- Routineentscheidungen werden unnötig an Senior-Teammitglieder eskaliert
- Die Arbeit stoppt oder verlangsamt sich erheblich, wenn Schlüsselwissensträger nicht verfügbar sind
- Teammitglieder äußern mangelndes Vertrauen, Entscheidungen ohne Konsultation zu treffen

## Symptoms ▲

- [Engpassbildung](engpassbildung.md)
<br/>  Schlüsselwissensträger werden zu Engpässen, während Teammitglieder sich in einer Warteschlange für ihre Anleitung anstellen.
- [Verringerte Teamproduktivität](verringerte-teamproduktivitaet.md)
<br/>  Die Arbeit stockt, wenn Wissensträger nicht verfügbar sind, was den Gesamtdurchsatz des Teams verringert.
- [Mentoren-Burnout](mentoren-burnout.md)
<br/>  Erfahrene Entwickler brennen durch ständige Unterbrechungen aus, um Fragen zu beantworten und Entscheidungen für andere zu treffen.
- [Single Points of Failure](single-points-of-failure.md)
<br/>  Kritisches Wissen, das bei wenigen Personen konzentriert ist, schafft Single Points of Failure für das Team.
- [Langsame Entwicklungsgeschwindigkeit](langsame-entwicklungsgeschwindigkeit.md)
<br/>  Die Entwicklung verlangsamt sich, weil abhängige Teammitglieder nicht fortfahren können, ohne Wissensträger zu konsultieren.

## Causes ▼

- [Zusammenbruch des Wissensaustauschs](zusammenbruch-des-wissensaustauschs.md)
<br/>  Unwirksame Wissensaustausch-Mechanismen zwingen Teammitglieder, sich auf Personen statt auf Dokumentation zu verlassen.
- [Implizites Wissen](implizites-wissen.md)
<br/>  Wenn kritisches Wissen nur in den Köpfen von Menschen existiert, müssen sich andere auf diese Personen verlassen, um darauf zuzugreifen.
- [Informationsverfall](informationsverfall.md)
<br/>  Wenn Dokumentation veraltet und unzuverlässig wird, müssen sich Teammitglieder auf Personen statt auf Dokumente verlassen.

## Detection Methods ○

- **Fragenabhängigkeits-Tracking:** Beobachtung, wie oft Teammitglieder Fragen stellen, die sie selbstständig beantworten können sollten
- **Entscheidungseskalationsanalyse:** Nachverfolgung, welche Arten von Entscheidungen eskaliert werden und ob die Eskalation angemessen ist
- **Häufigkeit von Arbeitsblockaden:** Messung, wie oft Arbeit blockiert wird, während auf bestimmte Personen gewartet wird
- **Unabhängigkeitsbewertung:** Bewertung der Fähigkeit von Teammitgliedern, autonom an altersgemäßen Aufgaben zu arbeiten
- **Auswirkung der Verfügbarkeit von Wissensträgern:** Bewertung, wie sich die Teamproduktivität ändert, wenn Schlüsselwissensträger nicht verfügbar sind

## Examples

Ein Entwickler, der seit acht Monaten im Team ist, stellt dem Senior-Architekten immer noch grundlegende Fragen zu Datenbankschemadesign, API-Endpunkten und Geschäftslogik, die inzwischen gut in seinem Verständnis liegen sollten. Trotz Zugang zu Dokumentation und früheren Codebeispielen sucht er durchgängig Bestätigung für Routineentscheidungen und Implementierungsansätze. Diese Abhängigkeit bedeutet, dass der Architekt täglich 2-3 Stunden damit verbringt, Fragen zu beantworten, die durch Dokumentation oder Experimentieren gelöst werden könnten, während die Arbeit des abhängigen Entwicklers häufig ins Stocken gerät, während er auf Antworten wartet. Ein weiteres Beispiel betrifft ein Team, in dem Entwickler mittleren Levels Codeänderungen nicht deployen können, ohne dass ein Senior-Entwickler ihre Deployment-Skripte und Konfigurationsänderungen überprüft, selbst für Routineaktualisierungen. Diese Abhängigkeit schafft Deployment-Engpässe und hindert das Team daran, kontinuierliche Deployment-Praktiken zu implementieren, weil zu vielen Teammitgliedern das Vertrauen und Wissen fehlt, Deployments selbstständig zu handhaben.
