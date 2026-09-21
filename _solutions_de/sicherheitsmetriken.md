---
title: Sicherheitsmetriken
description: Definition, Erfassung und Bewertung von Metriken zur
  Quantifizierung des Sicherheitsstatus.
category:
- Security
- Management
problems:
- difficulty-quantifying-benefits
- invisible-nature-of-technical-debt
- monitoring-gaps
- quality-blind-spots
- insufficient-audit-logging
- poor-project-control
layout: solution
lang: de
en_slug: security-metrics
related_solutions:
- slug: security-relevant-metrics
  similarity: 0.95
- slug: security-monitoring
  similarity: 0.8
- slug: security-frameworks
  similarity: 0.8
- slug: security-audits
  similarity: 0.8
- slug: security-certification
  similarity: 0.8
- slug: threat-modeling
  similarity: 0.8
---

## Description

Sicherheitsmetriken sind definierte, regelmäßig erfasste Messungen — wie durchschnittliche Zeit bis zum Patch, Schwachstellendichte, Vorfallhäufigkeit und Falsch-Positiv-Raten —, die die Sicherheitslage einer Organisation von einem impliziten Eindruck in eine explizite, verfolgbare Größe verwandeln. Der Mechanismus hängt von konsistenter Erfassung über die Zeit ab, statt einer einmaligen Messung: Ein einzelner Datenpunkt sagt wenig, aber eine Trendlinie, die sich verbessernde Patch-Zeiten oder sich in bestimmten Komponenten konzentrierende Schwachstellendichte zeigt, offenbart, wo sich die Sicherheitslage echt ändert und wo Ressourcen tatsächlich benötigt werden, was durch Intuition allein schwer wahrzunehmen ist. Dies ist besonders wertvoll für Legacy-Systeme, weil das in altem Code eingebettete Sicherheitsrisiko standardmäßig unsichtbar zu sein neigt — niemand erlebt eine langsam alternde Authentifizierungsbibliothek als Ereignis, wie sie einen Ausfall erlebt —, sodass ohne Metriken das Argument für die Investition in Legacy-Sicherheitsverbesserung keine Evidenz zum Aufbauen hat und gegen Arbeit mit sichtbarerer, unmittelbarerer Rendite verliert. Dieses unsichtbare Risiko in eine Zahl zu verwandeln, etwa durch das Zeigen, dass Legacy-Komponenten eine mehrfach längere Patch-Zeit als neuere haben, gibt Teams ein konkretes Artefakt, um Budget zu rechtfertigen und Behebung zu priorisieren, und lässt Fortschritt demonstriert statt nur behauptet werden. Das entsprechende Risiko ist, dass schlecht gewählte Metriken Aktivität statt Ergebnis messen und manipuliert werden können — Befunde zu schließen, ohne sie tatsächlich zu beheben, zum Beispiel —, sodass die gewählten Metriken echte Risikoreduktion widerspiegeln müssen, statt nur bequem zu erfassen zu sein.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Definieren Sie bedeutsame Sicherheitsmetriken, ausgerichtet an organisatorischer Risikobereitschaft und Sicherheitszielen
- Erfassen Sie Metriken wie durchschnittliche Zeit bis zum Patch, Schwachstellendichte, Vorfallhäufigkeit und Falsch-Positiv-Raten
- Automatisieren Sie Metrikenerfassung durch Integration mit Sicherheitswerkzeugen, Issue-Trackern und Überwachungssystemen
- Erstellen Sie Dashboards, die Sicherheitsmetriken sowohl technischen Teams als auch exekutiven Stakeholdern präsentieren
- Etablieren Sie Basislinien und setzen Sie Verbesserungsziele für Schlüsselsicherheitsindikatoren
- Überprüfen Sie Metriken regelmäßig in Sicherheits-Governance-Meetings und nutzen Sie sie zur Ressourcenallokation
- Verfolgen Sie Trends über die Zeit, statt sich auf einzelne Datenpunkte zu konzentrieren

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Ermöglicht datengetriebene Sicherheitsentscheidungen und Investitionspriorisierung
- Macht die Sicherheitslage für nicht-technische Stakeholder sichtbar und kommunizierbar
- Identifiziert Trends, die sich verbessernde oder verschlechternde Sicherheit signalisieren, bevor Vorfälle auftreten
- Unterstützt Verantwortlichkeit, indem Sicherheitsperformance messbar wird

**Kosten und Risiken:**
- Schlecht gewählte Metriken können kontraproduktives Verhalten anreizen (z. B. Befunde schließen, ohne sie zu beheben)
- Metrikenerfassung fügt bereits belasteten Legacy-System-Teams Overhead hinzu
- Sicherheitsmetriken können falsches Vertrauen erzeugen, wenn sie Aktivität statt Wirksamkeit messen
- Legacy-Systemen könnte die für automatisierte Metrikenerfassung nötige Instrumentierung fehlen

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Technologieunternehmen, das damit kämpfte, sein Sicherheitsbudget für Legacy-System-Wartung zu rechtfertigen, begann vier Schlüsselmetriken zu verfolgen: durchschnittliche Tage bis zum Patchen kritischer Schwachstellen, Anzahl offener Befunde hoher Schwere, Prozentsatz des von Sicherheitstests abgedeckten Codes und durchschnittliche Zeit bis zur Erkennung von Sicherheitsvorfällen. Nach sechs Monaten Verfolgung zeigten die Daten, dass ihre Legacy-Systeme eine durchschnittliche Patch-Zeit von 67 Tagen hatten, verglichen mit 12 Tagen für neuere Systeme, was konkrete Rechtfertigung für eine dedizierte Legacy-Sicherheitsverbesserungsinitiative lieferte. Das Metriken-Dashboard offenbarte außerdem, dass 80 % ihrer offenen Befunde sich in zwei Legacy-Komponenten konzentrierten, was fokussierte Behebung ermöglichte.
