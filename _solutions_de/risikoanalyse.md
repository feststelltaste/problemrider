---
title: Risikoanalyse
description: Identifikation, Bewertung und Behandlung von Risiken.
category:
- Security
- Management
problems:
- modernization-strategy-paralysis
- fear-of-change
- deployment-risk
- regulatory-compliance-drift
- invisible-nature-of-technical-debt
- high-technical-debt
- quality-blind-spots
- difficulty-quantifying-benefits
layout: solution
lang: de
en_slug: risk-analysis
related_solutions:
- slug: threat-modeling
  similarity: 0.8
- slug: security-architecture-analysis
  similarity: 0.8
- slug: requirements-analysis
  similarity: 0.8
- slug: emulation
  similarity: 0.75
- slug: security-relevant-metrics
  similarity: 0.75
- slug: functional-gap-analysis
  similarity: 0.75
---

## Description

Risikoanalyse ist der strukturierte Prozess, potenzielle Risiken für ein System zu identifizieren — Sicherheitsschwachstellen, betriebliche Schwächen, nicht unterstützte Technologie, angesammelte technische Schulden — und jedes nach Wahrscheinlichkeit und potenzieller Geschäftsauswirkung zu bewerten, sodass begrenzter Behebungsaufwand auf die Risiken gerichtet werden kann, die am meisten zählen, statt dünn über alles verteilt zu werden, was theoretisch schiefgehen könnte. Die Ausgabe ist typischerweise ein Risikoregister, das jedes Risiko benennt, bewertet, einen Eigentümer zuweist und seinen Milderungsstatus verfolgt, und verwandelt ein diffuses Unbehagen über ein Legacy-System in eine konkrete, priorisierte Arbeitsliste. Dies zählt in der Legacy-Modernisierung, weil die in einem alten System angesammelten Risiken üblicherweise zahlreich, schlecht dokumentiert und einzeln unquantifiziert sind — jeder spürt, dass das System riskant ist, aber wenige können artikulieren, welches spezifische Risiko am wahrscheinlichsten einen kostspieligen Vorfall verursacht, was es nahezu unmöglich macht, einen rationalen Investitionsfall zu rechtfertigen, um eines davon zuerst anzugehen. Eine strukturierte Risikoanalyse legt Muster offen, die Intuition allein übersieht, wie mehrere scheinbar unzusammenhängende Risiken, die alle auf eine einzige nicht unterstützte Middleware-Komponente zurückgehen, was ein diffuses Modernisierungsmandat in eine fokussierte, verteidigbare Priorität verwandelt. Da die Dokumentation von Legacy-Systemen oft unvollständig ist, beinhaltet Risikobewertung in diesem Kontext notwendigerweise echte Unsicherheit, und das Register selbst erfordert periodische Überprüfung, um nützlich zu bleiben, während sich das System und seine Umgebung weiterentwickeln.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Identifizieren und katalogisieren Sie alle bekannten Risiken im Legacy-System, einschließlich Sicherheitsschwachstellen, betrieblicher Schwächen und technischer Schulden
- Bewerten Sie jedes Risiko nach Wahrscheinlichkeit und potenzieller Auswirkung mittels eines strukturierten Bewertungs-Frameworks
- Priorisieren Sie Risiken und erstellen Sie ein Risikoregister, das jedes Risiko verantwortlichen Eigentümern und Milderungsplänen zuordnet
- Führen Sie regelmäßige Risiko-Review-Sitzungen mit Stakeholdern durch, um Bewertungen zu aktualisieren, während sich das System weiterentwickelt
- Nutzen Sie Risikoanalyse-Ergebnisse, um Modernisierungsprioritäten und Budgetallokationsentscheidungen anzutreiben
- Dokumentieren Sie akzeptierte Risiken mit klarer Begründung und überprüfen Sie sie periodisch
- Integrieren Sie Risikoanalyse in Change-Management-Prozesse für Legacy-System-Modifikationen

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Bietet datengetriebene Rechtfertigung für Sicherheits- und Modernisierungsinvestitionen
- Hilft Teams, begrenzte Ressourcen auf die Risiken mit der höchsten Auswirkung zu konzentrieren
- Schafft gemeinsames Verständnis von Systemschwachstellen über technische und geschäftliche Stakeholder hinweg
- Ermöglicht fundierte Risikoakzeptanzentscheidungen statt unerkannter Exposition

**Kosten und Risiken:**
- Risikobewertungen erfordern funktionsübergreifenden Input und können zeitaufwendig sein
- Die Quantifizierung von Risiken in Legacy-Systemen mit unvollständiger Dokumentation ist von Natur aus unsicher
- Risikoregister werden veraltet, wenn sie nicht aktiv gepflegt und überprüft werden
- Übermäßiges Vertrauen auf Risikobewertungen kann falsche Präzision und fehlgeleitete Prioritäten erzeugen

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Versicherungsunternehmen betreute ein Schadensverarbeitungssystem, das auf einem Technologie-Stack aufgebaut war, der von keinem Anbieter mehr unterstützt wurde. Eine strukturierte Risikoanalyse identifizierte 23 distinkte Risiken, reichend von ungepatchten bekannten Schwachstellen bis zu Single Points of Failure in der Bereitstellungspipeline. Durch die Bewertung jedes Risikos nach Auswirkung und Wahrscheinlichkeit identifizierte das Team, dass die drei höchsten Risiken alle mit derselben nicht unterstützten Middleware-Komponente zusammenhingen. Dies fokussierte den Modernisierungsaufwand darauf, zuerst diese einzelne Komponente zu ersetzen, statt eine vollständige Systemneuschreibung zu versuchen, was die gesamte Risikoexposition innerhalb eines einzigen Quartals um 60 % reduzierte.
