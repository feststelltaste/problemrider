---
title: Nutzerzentriertes Design
description: Einbeziehung von Nutzerbedürfnissen, -erwartungen und
  -fähigkeiten von Anfang an.
category:
- Requirements
- Business
quality_tactics_url: https://qualitytactics.de/en/usability/user-centered-design/
problems:
- poor-user-experience-ux-design
- user-frustration
- user-confusion
- user-trust-erosion
- negative-user-feedback
- customer-dissatisfaction
- negative-brand-perception
- feature-gaps
- shadow-systems
- competitive-disadvantage
- high-client-side-resource-consumption
- high-resource-utilization-on-client
- inefficient-frontend-code
layout: solution
lang: de
en_slug: user-centered-design
related_solutions:
- slug: consistent-user-interface
  similarity: 0.85
- slug: intuitive-navigation
  similarity: 0.8
- slug: usability-tests
  similarity: 0.8
- slug: cognitive-load-minimization
  similarity: 0.8
- slug: user-stories
  similarity: 0.8
- slug: a-b-testing
  similarity: 0.8
---

## Description

Nutzerzentriertes Design baut eine Schnittstelle um echte Nutzerforschung herum — Interviews, Journey Maps, Usability-Testing — statt um die technischen Einschränkungen und Entwicklerannahmen, die sie ursprünglich formten, was genau ist, wie die meisten Legacy-Schnittstellen überhaupt erst entstanden. Legacy-Systeme im Besonderen neigen dazu, ihr Datenbankschema oder das mentale Modell des ursprünglichen Entwicklers zu spiegeln, statt den Workflow der Person, die sie tatsächlich nutzt, und die dadurch entstehende Lücke treibt Nutzer zu Schatten-Tabellenkalkulationen und persönlichen Workarounds, von deren Existenz das Entwicklungsteam möglicherweise nicht einmal weiß, bis jemand endlich fragt. Die Etablierung einer regelmäßigen Forschungs- und Testtaktung, und die Priorisierung von Fixes dort, wo sich Nutzertraffic und Support-Tickets tatsächlich konzentrieren, verwandelt UX-Verbesserung in eine datengetriebene, inkrementelle Praxis statt eines einzelnen spekulativen Redesigns — obwohl es Forschungsfähigkeit erfordert, die viele legacy-fokussierte Teams noch nicht haben, und es vorübergehend die Inkonsistenz zwischen den bereits modernisierten Teilen des Systems und den noch wartenden Teilen erhöhen kann.

## How to Apply ◆

> Legacy-Systeme leiden häufig unter Schnittstellen, die um technische Einschränkungen und Entwicklerannahmen herum gestaltet wurden, statt um echte Nutzerbedürfnisse. Die Einführung nutzerzentrierter Design-Praktiken in Legacy-Modernisierungsanstrengungen stellt sicher, dass Verbesserungen echte statt eingebildeter Probleme adressieren.

- Führen Sie strukturierte Nutzerforschung durch, bevor Sie irgendeine Legacy-Schnittstelle neu gestalten. Interviewen Sie mindestens fünf repräsentative Nutzer pro Rolle, um ihre tatsächlichen Workflows, Schmerzpunkte und Workarounds zu verstehen. Legacy-Systeme haben oft Nutzer, die sich über Jahre an die Eigenheiten des Systems angepasst haben, und ihre Einblicke offenbaren sowohl was bewahrt werden muss als auch was sich ändern muss.
- Erstellen Sie Nutzer-Journey-Maps, die dokumentieren, wie Nutzer derzeit ihre Ziele mit dem Legacy-System erreichen, einschließlich der manuellen Workarounds, Schatten-Tabellenkalkulationen und externen Werkzeuge, die sie nutzen, um Systemmängel auszugleichen. Diese Karten legen den wahren Umfang der UX-Probleme offen, dessen sich das Entwicklungsteam möglicherweise nicht einmal bewusst ist.
- Etablieren Sie eine regelmäßige Usability-Testing-Taktung, bei der echte Nutzer versuchen, repräsentative Aufgaben mit dem System abzuschließen. Bei Legacy-Systemen ist dies oft aufschlussreich, weil Entwickler, die sich an die Schnittstelle gewöhnt haben, die Verwirrung nicht sehen können, die neue oder gelegentliche Nutzer erleben.
- Wenden Sie Prinzipien der progressiven Offenlegung an, wenn Sie komplexe Legacy-Schnittstellen modernisieren: Zeigen Sie Nutzern nur die Kontrollen und Informationen, die für ihre aktuelle Aufgabe relevant sind, und bieten Sie Zugang zu erweiterten Features durch bewusste Erkundung, statt die primäre Schnittstelle zu überwältigen.
- Implementieren Sie einen In-App-Feedback-Mechanismus, der Nutzern erlaubt, Probleme und Frustrationen direkt im Kontext zu melden, wobei der spezifische Bildschirm, Workflow-Schritt und die Aktion erfasst werden, die das Problem verursachte. Dies bietet kontinuierliches, reibungsarmes Feedback, das weit umsetzbarer ist als periodische Umfragen.
- Pflegen Sie ein Designsystem oder einen Style Guide, der Konsistenz über alle Teile der Anwendung sicherstellt, einschließlich sowohl modernisierter als auch noch nicht modernisierter Abschnitte. Inkonsistenz zwischen verschiedenen Teilen des Systems ist eine Hauptquelle von Nutzerverwirrung in Legacy-Anwendungen, die inkrementelle Verbesserung durchlaufen.
- Priorisieren Sie UX-Verbesserungen in den Bereichen der Anwendung mit dem höchsten Nutzertraffic und den meisten Support-Tickets, statt ein vollständiges Redesign zu versuchen. Datengetriebene Priorisierung stellt sicher, dass begrenzte Designressourcen die Probleme adressieren, die die meisten Nutzer betreffen.
- Beziehen Sie Barrierefreiheitsstandards (WCAG 2.1 AA Minimum) in jede UX-Verbesserung ein, nicht als nachträglichen Gedanken, sondern als Kern-Designbeschränkung. Legacy-Systemen fehlen häufig Barrierefreiheitsfeatures, und sie während der Modernisierung nachzurüsten ist weit günstiger als sie separat zu adressieren.

## Tradeoffs ⇄

> Nutzerzentriertes Design verwandelt Legacy-Systeme von innen-nach-außen (für Entwickler gebaut) zu außen-nach-innen (für Nutzer gebaut), erfordert aber anhaltende Investition in Forschungs- und Designfähigkeiten, die vielen legacy-fokussierten Organisationen fehlen.

**Vorteile:**

- Adressiert direkt die Grundursache von Nutzerfrustration, negativem Feedback und Kundenunzufriedenheit, indem Schnittstellen basierend auf validierten Nutzerbedürfnissen statt Entwicklerannahmen gestaltet werden.
- Reduziert die Proliferation von Schattensystemen, indem offizielle Systeme genuin nützlich für ihre beabsichtigten Nutzer gemacht werden, was die Motivation für Workarounds beseitigt.
- Bietet messbare Geschäftsauswirkung durch verbesserte Aufgabenabschlussraten, reduzierte Support-Ticket-Volumina und gesteigertes Nutzerengagement — Metriken, die fortgesetzte Investition in UX-Verbesserungen rechtfertigen.
- Baut Nutzervertrauen inkrementell auf, indem demonstriert wird, dass die Organisation auf Nutzerfeedback hört und danach handelt, was der Erosion des Vertrauens durch Jahre schlechter Erfahrungen entgegenwirkt.
- Verhindert Feature-Lücken, indem Funktionalitätsanforderungen mit echten Nutzern vor der Entwicklung validiert werden, was sicherstellt, dass das, was gebaut wird, dem entspricht, was Nutzer tatsächlich brauchen.

**Kosten und Risiken:**

- Nutzerforschung und Usability-Testing erfordern dedizierte Zeit und Ressourcen, die mit Feature-Lieferung konkurrieren, und Organisationen, die daran gewöhnt sind, ohne Nutzerinput auszuliefern, könnten sich gegen die wahrgenommene Verlangsamung sträuben.
- Legacy-Systemnutzer, die sich über Jahre an bestehende Workflows angepasst haben, könnten sich anfänglich gegen Schnittstellenänderungen sträuben, selbst wenn das neue Design objektiv besser ist, was sorgfältiges Change-Management erfordert.
- Inkrementelle UX-Verbesserungen in einem Legacy-System können vorübergehende Inkonsistenzen zwischen modernisierten und nicht modernisierten Abschnitten schaffen, was Nutzerverwirrung kurzfristig erhöhen könnte.
- Starke Investition in UX-Design für ein System, das möglicherweise ersetzt wird, schafft eine Spannung zwischen der Verbesserung der aktuellen Erfahrung und der Planung für eine zukünftige Plattform.
- Organisationen ohne Design-Expertise müssen entweder UX-Fachleute einstellen oder bestehendes Personal schulen, beides erfordert Investition, bevor sich Vorteile materialisieren.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie nutzerzentrierte Design-Praktiken die spezifischen UX-Herausforderungen adressieren, die in Legacy-Systemen gefunden werden.

Ein über fünfzehn Jahre gebautes Krankenhausinformationssystem hat eine Patiententerminierungsschnittstelle, die von Pflegekräften verlangt, durch sieben Bildschirme zu navigieren, um eine einzige Buchung abzuschließen. Die Schnittstelle wurde von Entwicklern gestaltet, um die Datenbankstruktur zu spiegeln, statt den klinischen Workflow. Nach Shadowing-Sitzungen mit Pflegepersonal entdeckt das UX-Team, dass Pflegekräfte aufwendige papierbasierte Checklisten entwickelt haben, um sich die Bildschirm- und Feldsequenz zu merken. Das Team gestaltet den Terminierungsablauf in einen Einseiten-Wizard um, der dem tatsächlichen klinischen Workflow folgt, mit progressiver Offenlegung für Ausnahmefälle. Support-Tickets im Zusammenhang mit Terminierungsfehlern sinken innerhalb von drei Monaten um 65 %, und das Pflegepersonal, das zuvor Schatten-Tabellenkalkulationen zur Verfolgung von Buchungen pflegte, kehrt zur Nutzung des offiziellen Systems zurück.

Das Schadensbearbeitungssystem eines Versicherungsunternehmens erzeugt anhaltendes negatives Nutzerfeedback wegen inkonsistenter Terminologie über verschiedene Abschnitte hinweg — dasselbe Konzept wird je nach Modul als "Schadennummer", "Fall-ID" oder "Referenzcode" bezeichnet. Ein Design-Audit deckt 47 Instanzen inkonsistenter Terminologie auf. Das Team etabliert ein Designsystem mit einem standardisierten Glossar, wendet es systematisch während regulärer Wartungsarbeit an und implementiert kontextuelle Hilfe-Tooltips, die Begriffe erklären, die Nutzer verwirrend finden. Über sechs Monate sinkt das Anrufvolumen beim internen Helpdesk um 40 %, und die Einarbeitungszeit für neue Mitarbeitende im Schadenssystem sinkt von drei Wochen auf eine Woche. Das konsistente Vokabular reduziert auch Dateneingabefehler, bei denen Sachbearbeiter Informationen in falsche Felder eingaben, weil sie die Beschriftungen missverstanden.
