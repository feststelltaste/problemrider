---
title: Analyse der Architekturkonformität
description: Prüfung der Übereinstimmung der Softwarearchitektur mit definierten
  Architekturprinzipien.
category:
- Architecture
problems:
- stagnant-architecture
- high-coupling-low-cohesion
- architectural-mismatch
- inconsistent-codebase
- ripple-effect-of-changes
- tight-coupling-issues
- monolithic-architecture-constraints
- technical-architecture-limitations
- circular-dependency-problems
layout: solution
lang: de
en_slug: architecture-conformity-analysis
related_solutions:
- slug: fitness-functions
  similarity: 0.8
- slug: architecture-reviews
  similarity: 0.8
- slug: static-analysis-and-linting
  similarity: 0.8
- slug: architecture-governance
  similarity: 0.75
- slug: architecture-decision-records
  similarity: 0.75
- slug: architecture-documentation
  similarity: 0.75
---

## Description

Analyse der Architekturkonformität prüft automatisch, ob die tatsächlichen Abhängigkeiten und Kommunikationsmuster einer Codebasis mit einer Reihe beabsichtigter Architekturregeln übereinstimmen — etwa welche Schichten welche aufrufen dürfen oder welche Module direkt kommunizieren dürfen — unter Nutzung von Werkzeugen wie ArchUnit, Structure101 oder Sonargraph, die diese Regeln als ausführbare Prüfungen kodieren können, die in Continuous Integration laufen. In Legacy-Systemen driften die ursprünglich designte Architektur und die tatsächlich im Code existierende Architektur über Jahre dringender Fixes und Abkürzungen stetig auseinander, bis die Diagramme in welcher Dokumentation auch immer noch existiert nicht mehr beschreiben, wie sich das System tatsächlich verhält, und niemand mit Sicherheit sagen kann, wie viele Verstöße gegen das ursprüngliche Design sich angehäuft haben. Diese Drift sichtbar zu machen ist der notwendige erste Schritt, bevor sie angegangen werden kann: Durch die Kodierung der beabsichtigten Regeln als automatisierte Prüfungen werden zuvor unsichtbare Verstöße — wie Präsentationscode, der direkt Datenbank-Repositories aufruft und dabei eine ganze Business-Schicht umgeht — zu einer expliziten, zählbaren Metrik statt eines vagen Gefühls, dass die Architektur nicht mehr das ist, was sie einmal war. Weil eine Legacy-Codebasis ohne Durchsetzungshistorie typischerweise bereits eine große Anzahl von Verstößen enthält, wird die Konformitätsanalyse üblicherweise mit einer Baseline akzeptierter Verstöße und einem Ziel für schrittweise Reduktion eingeführt, statt eines harten Gates, das jede weitere Entwicklung blockieren würde. Das kontinuierliche Ausführen dieser Prüfungen verhindert außerdem, dass neue Verstöße hinzugefügt werden, während das Team den bestehenden Rückstand abarbeitet, was Konformitätsanalyse zu einem dauerhaften Prozess statt einer einmaligen Bereinigung macht, und es ist oft die Entdeckung, dass Legacy-Schichtgrenzen doch vollständig durchsetzbar waren, die nachfolgende Modernisierungsschritte wie den Austausch einer Datenzugriffsschicht erschließt.

## How to Apply ◆

> In Legacy-Systemen weicht die tatsächliche Architektur fast immer von der beabsichtigten Architektur ab — Konformitätsanalyse macht diese Drift sichtbar und handhabbar.

- Definieren Sie die beabsichtigten Architekturregeln explizit (Schichtabhängigkeiten, Modulgrenzen, erlaubte Kommunikationsmuster) in einem Format, das automatisch geprüft werden kann.
- Nutzen Sie Architekturanalysewerkzeuge (wie ArchUnit, Structure101 oder Sonargraph), um Verstöße gegen Architekturregeln in der bestehenden Codebasis automatisch zu erkennen.
- Führen Sie Konformitätsprüfungen als Teil der Continuous-Integration-Pipeline aus, sodass neue Verstöße erfasst werden, bevor sie gemergt werden.
- Beginnen Sie mit den kritischsten Architekturgrenzen — wie der Trennung zwischen Domänenlogik und Infrastruktur —, statt zu versuchen, alle Regeln auf einmal in einer Legacy-Codebasis mit vielen bestehenden Verstößen durchzusetzen.
- Erstellen Sie eine Baseline bekannter Verstöße und verfolgen Sie die Reduktion über die Zeit, indem Sie Konformitätsverbesserung als messbares Modernisierungsziel behandeln.
- Überprüfen Sie Konformitätsanalyseergebnisse in Architektur-Review-Meetings, um zu entscheiden, welche Verstöße behoben, welche vorübergehend akzeptiert und welche Regeln überarbeitet werden sollen.

## Tradeoffs ⇄

> Konformitätsanalyse verhindert Architekturerosion, erfordert aber klare Regeln und Team-Zustimmung, um effektiv zu sein.

**Vorteile:**

- Macht Architekturverstöße sichtbar, bevor sie sich zu strukturellem Verfall anhäufen, der teuer umzukehren ist.
- Bietet objektive, messbare Kriterien für Architekturqualität statt sich auf subjektive Einschätzungen zu verlassen.
- Verhindert, dass neue Entwicklung die Architektur weiter verschlechtert, während das Team an der Behebung bestehender Verstöße arbeitet.
- Unterstützt schrittweise Legacy-Modernisierung, indem verfolgt wird, wie sich die Architekturkonformität über die Zeit verbessert.

**Kosten und Risiken:**

- Die Definition von Regeln für ein Legacy-System, das nie explizite Architekturrichtlinien hatte, erfordert erheblichen Vorabaufwand und architektonisches Urteilsvermögen.
- Zu viele oder übermäßig strenge Regeln können Entwickler frustrieren und zu Workarounds führen, die die Prüfungen umgehen.
- Konformitätsanalysewerkzeuge können Konfigurationsaufwand erfordern und möglicherweise nicht alle in Legacy-Systemen genutzten Sprachen oder Frameworks unterstützen.
- Die alleinige Fokussierung auf strukturelle Konformität kann höherstufige Architekturprobleme wie unpassende Technologiewahlen oder fehlende Qualitätsattribute übersehen.

## How It Could Be

> Das folgende Szenario zeigt, wie Konformitätsanalyse Architekturerosion aufdeckt und verhindert.

Ein Softwareunternehmen, das eine 10 Jahre alte Unternehmensanwendung pflegte, entdeckte durch Konformitätsanalyse, dass seine beabsichtigte Drei-Schichten-Architektur (Präsentation, Business, Datenzugriff) 340 Verstöße aufwies, bei denen Präsentationsschicht-Klassen direkt auf Datenbank-Repositories zugriffen und dabei die Business-Schicht vollständig umgingen. Diese Abkürzungen hatten sich über Jahre dringender Bugfixes und Feature-Anfragen angehäuft. Das Team konfigurierte ArchUnit-Regeln, um neue Verstöße zu verhindern, und etablierte ein „Verstoßbudget", das sich jedes Quartal um 10 % verringerte. Über 18 Monate sank die Anzahl der Verstöße von 340 auf 45, und die verbleibenden Verstöße waren dokumentierte Ausnahmen mit expliziter Begründung. Die durchgesetzten Schichtgrenzen machten es möglich, die Datenzugriffsschicht durch ein neues ORM zu ersetzen, ohne den Präsentationscode zu berühren — eine Änderung, die vor der Konformitätsarbeit unmöglich gewesen wäre.
