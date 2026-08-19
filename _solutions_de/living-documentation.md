---
title: Living Documentation
description: Aktuelle und leicht zugängliche Dokumentation als integraler
  Bestandteil der Entwicklung.
category:
- Communication
- Process
problems:
- poor-documentation
- information-decay
- legacy-system-documentation-archaeology
- implicit-knowledge
- knowledge-silos
- difficult-developer-onboarding
- tacit-knowledge
- unclear-documentation-ownership
- accumulated-decision-debt
- duplicated-research-effort
- extended-research-time
- information-fragmentation
- knowledge-sharing-breakdown
- team-churn-impact
- incomplete-knowledge
layout: solution
lang: de
en_slug: living-documentation
related_solutions:
- slug: documentation-as-code
  similarity: 0.85
- slug: architecture-documentation
  similarity: 0.8
- slug: documentation-of-compatibility-requirements
  similarity: 0.75
- slug: runbooks
  similarity: 0.75
- slug: pattern-language
  similarity: 0.75
- slug: architecture-decision-records
  similarity: 0.75
---

## Description

Living Documentation ist Dokumentation, die aus dem System, das sie beschreibt, generiert, in es eingebettet oder kontinuierlich gegen es verifiziert wird, sodass sie sich automatisch zusammen mit dem Code weiterentwickelt, statt in dem Moment davon abzudriften, in dem sie geschrieben wird. In der Praxis bedeutet dies, Dokumentation als versionskontrollierten Text neben dem Code zu speichern, den sie dokumentiert, API-Referenzen direkt aus Code-Annotationen oder OpenAPI-Spezifikationen zu generieren, Architecture Decision Records in dem Moment zu schreiben, in dem Entscheidungen getroffen werden, statt im Nachhinein, und ausführbare Spezifikationen zu nutzen, die einen Build in dem Moment fehlschlagen lassen, in dem das beschriebene Verhalten und das tatsächliche Verhalten auseinandergehen. Legacy-Systeme sind die Umgebung, in der konventionelle Dokumentation am härtesten scheitert: Wikis und Word-Dokumente werden einmal während einer anfänglichen Projektphase geschrieben und dann aufgegeben, während sich das System selbst jahrelang weiter ändert, sodass die Dokumentation zu dem Zeitpunkt, an dem ein neuer Pfleger sie braucht, ein System beschreibt, das nicht mehr existiert, und das daraus resultierende Misstrauen bedeutet, dass Menschen aufhören, sie zu konsultieren und zu aktualisieren, was den Verfall beschleunigt. Living Documentation durchbricht diesen Zyklus, indem der manuelle Synchronisationsschritt vollständig entfernt wird — weil die Dokumentation entweder mechanisch aus dem Code abgeleitet oder aktiv durch Tests erzwungen wird, kann sie nicht still veralten, ohne dass der Build selbst die Diskrepanz signalisiert. Dies zählt am meisten in der Legacy-Modernisierungsarbeit, wo institutionelles Wissen darüber, warum sich das System so verhält, wie es sich verhält, oft in wenigen Personen konzentriert ist, die gehen könnten, und wo neue Teammitglieder sonst keinen verlässlichen Weg haben, aktuellen, genauen Kontext von Jahre alten, überholten Notizen zu unterscheiden.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Speichern Sie Dokumentation neben dem Code in Versionskontrolle, damit sie sich mit dem System weiterentwickelt
- Nutzen Sie ausführbare Spezifikationen (z. B. BDD-artige Tests), die sowohl als Dokumentation als auch als Verifikation dienen
- Generieren Sie API-Dokumentation aus Code-Annotationen oder OpenAPI-Spezifikationen, um sie immer aktuell zu halten
- Übernehmen Sie Architecture Decision Records (ADRs), um die Begründung hinter wichtigen Designentscheidungen zu erfassen
- Integrieren Sie Dokumentationsprüfungen in die CI-Pipeline, um veraltete oder defekte Referenzen zu erkennen
- Ersetzen Sie statische Wiki-Seiten durch Documentation-as-Code-Ansätze, die in Pull Requests überprüft werden
- Beginnen Sie mit den Bereichen des Legacy-Systems, die die meiste Verwirrung oder Onboarding-Reibung verursachen

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Dokumentation bleibt genau, weil sie als Teil des normalen Entwicklungsworkflows aktualisiert wird
- Verringert die Onboarding-Zeit, indem auffindbares, aktuelles Systemwissen bereitgestellt wird
- Ausführbare Spezifikationen bieten sowohl Dokumentation als auch Regressionsschutz
- Versionskontrollierte Dokumentation bewahrt die Historie und ermöglicht Blame-Tracking

**Kosten und Risiken:**
- Erfordert kulturellen Wandel: Teams müssen Dokumentation als erstrangigen Liefergegenstand behandeln
- Anfänglicher Aufwand, die Infrastruktur zu etablieren und bestehende Dokumentation zu migrieren
- Ausführbare Spezifikationen fügen Pflegeaufwand hinzu, wenn sich Systemverhalten häufig ändert
- Risiko von Dokumentations-Aufblähung ohne Kuratierungs- oder Bereinigungsdisziplin

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Versicherungsunternehmen erbte ein Legacy-Schadenbearbeitungssystem, dessen einzige Dokumentation eine Reihe von Word-Dokumenten war, zuletzt vor fünf Jahren aktualisiert. Neue Entwickler verbrachten Wochen damit, Kollegen zu fragen, um Geschäftsregeln zu verstehen. Das Team begann, ADRs für jede bedeutende Änderung zu schreiben, fügte Javadoc-generierte API-Referenzen hinzu und führte Cucumber-Szenarien ein, die den Schadenworkflow in Geschäftssprache beschrieben. Innerhalb von sechs Monaten sank die Onboarding-Zeit von drei Wochen auf eine, und die Cucumber-Szenarien fingen mehrere Fälle ab, in denen die Dokumentation dem tatsächlichen Systemverhalten widersprach, was zu wichtigen Bug-Entdeckungen führte.
