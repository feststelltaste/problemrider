---
title: Incident Management
description: Strukturierter Prozess zum Umgang mit Störungen und Ausfällen.
category:
- Process
- Operations
problems:
- constant-firefighting
- slow-incident-resolution
- system-outages
- communication-breakdown
- poorly-defined-responsibilities
- knowledge-silos
- high-defect-rate-in-production
layout: solution
lang: de
en_slug: incident-management
related_solutions:
- slug: security-incident-handling
  similarity: 0.85
- slug: runbooks
  similarity: 0.85
- slug: root-cause-analysis
  similarity: 0.85
- slug: incident-response-measures
  similarity: 0.8
- slug: chaos-engineering
  similarity: 0.8
- slug: secure-software
  similarity: 0.8
---

## Description

Incident Management ist ein definierter, wiederholbarer Prozess zum Erkennen, Reagieren auf und Lernen von betrieblichen Störungen, aufgebaut um explizite Schweregrade, eine benannte Incident-Commander-Rolle, vorbereitete Kommunikationskanäle und dokumentierte Runbooks für bekannte Fehlermodi. Legacy-Systeme neigen dazu, Vorfälle anzusammeln, die ad hoc gehandhabt werden — gelöst von wem auch immer gerade verfügbar ist, mit Wissen, das nur im Kopf dieser Person existiert —, weil die Systeme jedem formalen Vorfallprozess vorausgehen und die institutionelle Gewohnheit, Ausfälle zu dokumentieren, nie etabliert wurde. Struktur um diese Aktivität einzuführen bewirkt zweierlei gleichzeitig: Es verkürzt die Zeit bis zur Lösung bei jedem einzelnen Vorfall, indem Entscheidungsverzögerungen und Rollenmehrdeutigkeit in einem stressreichen Moment beseitigt werden, und es wandelt jeden Vorfall in eine dauerhafte Quelle organisatorischen Lernens um durch schuldfreie Post-Incident-Reviews, die über die Zeit verfolgt werden, statt vergessen zu werden, sobald das unmittelbare Feuer gelöscht ist. Dieser zweite Effekt ist besonders wertvoll für Legacy-Systeme, wo dieselbe Handvoll Grundursachen oft für einen großen Anteil wiederkehrender Vorfälle verantwortlich ist; ein konsistenter Review-Prozess ist es, was dieses Muster ans Licht bringt, statt jedes Vorkommnis als isoliertes, unzusammenhängendes Ereignis zu behandeln. Die Kosten dieser Struktur sind Prozess-Overhead, der kontraproduktiv werden kann, wenn Verfahren zu starr sind, um einen Vorfall aufzunehmen, der nicht in die vordefinierten Kategorien passt, sodass der Prozess anpassungsfähig bleiben muss, während er formaler wird.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Definieren Sie Schweregrade mit klaren Kriterien und erwarteten Reaktionszeiten für Legacy-System-Vorfälle
- Etablieren Sie eine Incident-Commander-Rolle und klare Eskalationspfade für jeden Schweregrad
- Erstellen Sie Kommunikationsvorlagen und -kanäle, damit Stakeholder während Vorfällen zeitnahe Updates erhalten
- Bauen Sie Runbooks für bekannte Legacy-System-Fehlermodi mit Schritt-für-Schritt-Lösungsverfahren
- Führen Sie schuldfreie Post-Incident-Reviews durch, um gewonnene Erkenntnisse zu erfassen und Wiederholung zu verhindern
- Verfolgen Sie Vorfallmetriken (MTTR, MTTD, Häufigkeit nach Komponente), um systemische Probleme zu identifizieren
- Integrieren Sie Vorfallverfolgung mit der Monitoring- und Alarmierungsinfrastruktur des Legacy-Systems
- Üben Sie Vorfallreaktion durch regelmäßige Game-Day-Übungen

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Verringert die mittlere Lösungszeit durch strukturierte Reaktionsverfahren
- Verhindert Wissensverlust durch Dokumentation von Vorfallursachen und -lösungen
- Verringert Stress während Vorfällen durch klare Rollen und Kommunikationsprotokolle
- Schafft eine Feedback-Schleife, die systemische Zuverlässigkeitsverbesserungen antreibt

**Kosten und Risiken:**
- Prozess-Overhead kann die Reaktion verlangsamen, wenn Verfahren für schnelllebige Vorfälle zu starr sind
- Erfordert laufende Investition in Schulung und Dokumentationspflege
- Post-Incident-Reviews nehmen Zeit von der Feature-Entwicklung
- Übermäßig bürokratische Vorfallprozesse können davon abschrecken, kleinere Probleme zu melden

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein SaaS-Unternehmen kämpfte mit wiederkehrenden Ausfällen in seinem Legacy-Zahlungsverarbeitungssystem. Vorfälle wurden ad hoc von wem auch immer gerade verfügbar war gehandhabt, ohne konsistente Kommunikation an Stakeholder. Nach der Implementierung eines strukturierten Incident-Management-Prozesses mit definierten Schweregraden, benannten Incident Commandern und verpflichtenden Post-Incident-Reviews verringerte das Team seine mittlere Lösungszeit um 40 %. Wichtiger noch: Die Post-Incident-Reviews identifizierten drei wiederkehrende Grundursachen im Legacy-Code, die, einmal behoben, eine ganze Klasse von Produktionsvorfällen beseitigten.
