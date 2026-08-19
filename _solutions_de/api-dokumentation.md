---
title: API-Dokumentation
description: Detaillierte Beschreibung von Schnittstellen und ihrer Nutzung.
category:
- Communication
- Code
problems:
- poor-documentation
- poor-interfaces-between-applications
- difficult-developer-onboarding
- knowledge-gaps
- legacy-system-documentation-archaeology
- integration-difficulties
- stakeholder-developer-communication-gap
- implicit-knowledge
- communication-risk-outside-project
layout: solution
lang: de
en_slug: api-documentation
related_solutions:
- slug: api-first-design
  similarity: 0.8
- slug: documentation-as-code
  similarity: 0.8
- slug: architecture-documentation
  similarity: 0.8
- slug: api-first-development
  similarity: 0.8
- slug: contract-testing
  similarity: 0.75
- slug: living-documentation
  similarity: 0.75
---

## Description

API-Dokumentation ist eine strukturierte, detaillierte Beschreibung der Endpunkte, Anfrage- und Antwortformate, Fehlerbedingungen, Authentifizierungsanforderungen und Verhaltenseigenheiten einer Schnittstelle, idealerweise generiert aus oder validiert gegen die tatsächliche Definition der API, sodass sie nicht still aus der Synchronisation mit der Implementierung driften kann. In Legacy-Systemen ist das Fehlen solcher Dokumentation selten zufällig; es spiegelt üblicherweise eine Periode wider, in der die API für eine kleine, bekannte Menge interner Konsumenten gebaut wurde, die einfach direkt die ursprünglichen Entwickler fragen konnten, und das Wissen wurde nie aufgeschrieben, weil es nie nötig war. Diese informelle Vereinbarung bricht zusammen, sobald diese Entwickler gehen, die Konsumentenbasis über die Personen hinauswächst, die die API gebaut haben, oder Modernisierungsarbeit erfordert, dass andere Teams Verhalten verstehen, das nirgendwo spezifiziert wurde außer im Gedächtnis von jemandem. Die Rekonstruktion von Dokumentation für eine undokumentierte Legacy-API erfordert typischerweise das Reverse-Engineering tatsächlichen Verhaltens aus Client-Code, Integrationstests und Produktions-Traffic, da das Ziel ist zu erfassen, was das System tatsächlich tut — einschließlich seiner Eigenheiten und Fehlerbedingungen — statt einer idealisierten Beschreibung dessen, was es tun sollte. Einmal an einem zentralisierten, durchsuchbaren Ort veröffentlicht, verwandelt diese Dokumentation Integration in eine Self-Service-Aktivität statt eines Engpasses bei einer Handvoll Personen, und sie bringt häufig vergessene oder ungenutzte Endpunkte ans Licht, die sicher abgeschaltet werden können. Dokumentation, die nicht aktuell gehalten wird, ist schlimmer als keine, weil sie falsches Vertrauen schafft und genau die Entwickler in die Irre führt, denen sie helfen sollte, sodass die Praxis sich nur auszahlt, wenn sie als erforderlicher Schritt im API-Änderungsprozess statt als einmaliger Aufwand gepflegt wird.

## How to Apply ◆

> In Legacy-Systemen sind undokumentierte APIs eines der bedeutendsten Hindernisse für Integration, Modernisierung und Onboarding — was API-Dokumentation zu einer Voraussetzung für nachhaltige Veränderung macht.

- Beginnen Sie mit der Dokumentation der APIs, von denen die Modernisierungsbemühung am stärksten abhängt, mithilfe von Werkzeugen wie OpenAPI/Swagger, die interaktive Dokumentation aus API-Definitionen generieren.
- Reverse-Engineeren Sie Legacy-API-Verhalten durch Analyse bestehenden Client-Codes, Integrationstests und Produktions-Traffic-Protokolle, um tatsächliche Nutzungsmuster statt idealisierter Designs zu erfassen.
- Beinhalten Sie nicht nur Endpunktsignaturen, sondern auch Fehlerantworten, Ratenlimits, Authentifizierungsanforderungen, Datenformateigenheiten und bekannte Einschränkungen, die aktuell nur erfahrene Entwickler kennen.
- Generieren Sie Dokumentation aus Code oder API-Definitionen, wo immer möglich, um Dokumentation mit der tatsächlichen Implementierung synchron zu halten.
- Veröffentlichen Sie API-Dokumentation an einem zentralisierten, durchsuchbaren Ort, der für alle Teams zugänglich ist, die die APIs konsumieren, einschließlich externer Integrationspartner.
- Beinhalten Sie praktische Beispiele, die übliche Nutzungsmuster zeigen, besonders für komplexe Operationen, die mehrere API-Aufrufe in Sequenz erfordern.
- Etablieren Sie einen Dokumentations-Review-Schritt im API-Änderungsprozess, um sicherzustellen, dass Dokumentation aktuell bleibt, während sich APIs weiterentwickeln.

## Tradeoffs ⇄

> API-Dokumentation verringert Integrationsreibung und Wissensabhängigkeit dramatisch, erfordert aber anhaltenden Aufwand, um akkurat zu bleiben.

**Vorteile:**

- Verringert die Onboarding-Zeit für Entwickler, indem Self-Service-API-Lernen statt Mentoring durch erfahrene Teammitglieder geboten wird.
- Ermöglicht parallele Entwicklung, indem Teams erlaubt wird, sich mit APIs basierend auf Dokumentation zu integrieren, statt darauf zu warten, dass das API-Team für Fragen verfügbar ist.
- Bringt Inkonsistenzen und Designprobleme in Legacy-APIs ans Licht, die offensichtlich werden, wenn Verhalten explizit dokumentiert wird.
- Unterstützt Legacy-Systemmigration, indem eine klare Spezifikation geboten wird, der Ersatz-APIs entsprechen oder die sie übertreffen müssen.

**Kosten und Risiken:**

- Dokumentation, die vom tatsächlichen API-Verhalten abdriftet, ist schlimmer als keine Dokumentation, weil sie falsches Vertrauen und Debugging-Verwirrung schafft.
- Umfassende Dokumentation für eine große Legacy-API-Oberfläche kann ein erheblicher initialer Aufwand sein.
- Teams könnten sich dagegen wehren, APIs zu dokumentieren, die sie bald ersetzen möchten, was eine Lücke während der Übergangsperiode schafft.
- Automatisch generierte Dokumentation ohne narrativen Kontext mag technisch akkurat, aber für Entwickler, die Nutzungsmuster verstehen wollen, unhilfreich sein.

## How It Could Be

> Das folgende Szenario veranschaulicht die Auswirkung von API-Dokumentation auf Legacy-Systemintegration.

Ein Finanzdienstleistungsunternehmen hatte eine Legacy-Zahlungsverarbeitungs-API, die von 15 internen Anwendungen und 8 externen Partnern genutzt wurde. Die API hatte keine Dokumentation — alles Integrationswissen lebte in den Köpfen dreier Senior-Entwickler und in verstreuten E-Mail-Threads. Als zwei dieser Entwickler innerhalb von sechs Monaten gingen, wurde der verbleibende Entwickler zu einem Engpass für jede Integrationsfrage. Das Team investierte vier Wochen in die Dokumentation aller 120 Endpunkte mit OpenAPI-Spezifikationen, einschließlich Fehlercodes, Retry-Verhalten und Idempotenzanforderungen, die zuvor wiederkehrende Integrationsbugs verursacht hatten. Innerhalb von drei Monaten sank das Volumen der Integrations-Support-Anfragen um 70 %, und zwei neue Integrationspartner banden sich nur mithilfe der Dokumentation selbst ein. Die Dokumentationsbemühung offenbarte außerdem 23 Endpunkte, die vollständig ungenutzt waren, die das Team anschließend als veraltet markierte.
