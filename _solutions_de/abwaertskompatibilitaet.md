---
title: Abwärtskompatibilität
description: Gewährleistung, dass neue Versionen weiterhin mit bestehenden Clients,
  Daten und Integrationen funktionieren.
category:
- Architecture
- Dependencies
problems:
- breaking-changes
- api-versioning-conflicts
- integration-difficulties
- fear-of-breaking-changes
- regression-bugs
- ripple-effect-of-changes
- deployment-risk
- abi-compatibility-issues
- rapid-system-changes
layout: solution
lang: de
en_slug: backward-compatibility
related_solutions:
- slug: backward-compatible-apis
  similarity: 0.9
- slug: backward-compatible-data-formats
  similarity: 0.85
- slug: forward-compatibility
  similarity: 0.85
- slug: compatibility-as-error
  similarity: 0.8
- slug: compatibility-measurement
  similarity: 0.8
- slug: compatibility-testing
  similarity: 0.8
---

## Description

Abwärtskompatibilität ist die Eigenschaft, dass eine neue Version eines Systems, einer API oder eines Datenformats weiterhin bestehende Clients, Integrationen und gespeicherte Daten zufriedenstellt, ohne dass diese sich ändern müssen, erreicht durch die Behandlung bestehender Verträge als fest und die Weiterentwicklung nur durch additive, nicht-brechende Änderungen. Statt einer einzelnen Technik ist es eine explizite Beschränkung darüber, wie Veränderung geschehen darf: Neue Felder und Endpunkte können hinzugefügt werden, aber bestehende werden nicht geändert oder entfernt, und jede Änderung, die dies verletzen würde, wird aufgeschoben oder durch einen separaten, veralteten Pfad ausgeführt. Sie ist für Legacy-Systeme akut wichtig, weil solche Systeme typischerweise eine breite und oft unsichtbare Menge nachgelagerter Abhängiger anhäufen — andere interne Systeme, externe Partner, Batch-Jobs und Berichte —, gebaut über viele Jahre von Menschen, die nicht mehr da sind, um zu erklären, was von was abhängt, sodass eine gewöhnlich aussehende Änderung still Integrationen brechen kann, an die sich niemand mehr erinnert. Die Verpflichtung zur Abwärtskompatibilität verwandelt jedes Release in ein risikoarmes Ereignis für diese Abhängigen, die aktualisieren können, wann immer es passt, statt zu synchronisierten Migrationen gezwungen zu werden, auf direkte Kosten der Schnittstelle selbst: Verpflichtungen häufen sich an, Felder überleben ihren Nutzen, und manche architektonischen Verbesserungen werden unmöglich, ohne schließlich die Garantie zu brechen. Die spezifischen Instrumente, die dies praktikabel machen — Abwärtskompatible APIs, Abwärtskompatible Datenformate und Abwärtskompatible Schema-Migrationen — wenden dasselbe additive Prinzip auf verschiedenen Schichten desselben Legacy-Systems an.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Etablieren Sie Abwärtskompatibilität als explizite Anforderung für alle öffentlichen Schnittstellen und Datenformate
- Nutzen Sie nur additive Änderungen (neue Felder, neue Endpunkte) statt bestehende zu ändern oder zu entfernen
- Führen Sie bestehende Client-Test-Suiten gegen neue Versionen als Teil der CI-Pipeline aus
- Pflegen Sie Kompatibilitäts-Test-Suiten, die spezifisch verifizieren, dass alte Clients mit neuen Serverversionen funktionieren
- Führen Sie Feature-Flags ein, um neues Verhalten zusammen mit altem Verhalten während Übergangsperioden auszuliefern
- Dokumentieren Sie Kompatibilitätsgarantien und die Bedingungen, unter denen sie gebrochen werden dürfen

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Konsumenten können nach eigenem Zeitplan aktualisieren, ohne erzwungene Migrationen
- Verringert Deployment-Risiko, indem sichergestellt wird, dass bestehende Integrationen weiterhin funktionieren
- Baut Vertrauen bei externen API-Konsumenten und internen Teams gleichermaßen auf

**Kosten und Risiken:**
- Die Aufrechterhaltung von Abwärtskompatibilität kann API-Evolution und Innovation verlangsamen
- Angehäufte Kompatibilitätsbeschränkungen führen über die Zeit zu aufgeblähten Schnittstellen
- Manche architektonischen Verbesserungen sind ohne Bruch der Abwärtskompatibilität unmöglich
- Das Testen der vollständigen Matrix alter und neuer Kombinationen erhöht die CI-Kosten

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine Regierungsbehörde betrieb eine Datenaustauschplattform, die von 50 kommunalen Systemen genutzt wurde, von denen viele Software betrieben, die nur einmal jährlich aktualisiert wurde. Durch die Verpflichtung zu strikter Abwärtskompatibilität für das Austauschformat und das Hinzufügen neuer Felder als optionale Erweiterungen konnte die Behörde drei größere Plattform-Upgrades über zwei Jahre ausrollen, ohne dass irgendeine Kommune ihre Software ändern musste. Die wenigen Kommunen, die neue Felder übernahmen, gewannen zusätzliche Funktionalität, während andere ohne Störung weiter operierten.
